# 01_invoice_agent.py
# Invoice 3-way matching dashboard for finance teams.
# Decisions are deterministic Python/Pandas rules; Ollama is optional and never decides status.

import re
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Invoice 3-Way Matching", page_icon="🧾", layout="wide")
st.title("🧾 Invoice 3-Way Matching")
st.caption("Match invoices against purchase orders and goods receipts. Decisions are rule-based, not AI-generated.")

ALIASES = {
    "invoice_id": ["invoice_id", "invoice_no", "invoice_number", "inv_id", "bill_no"],
    "vendor": ["vendor", "vendor_name", "supplier", "supplier_name"],
    "po_id": ["po_id", "po_number", "purchase_order", "purchase_order_id"],
    "invoice_amount": ["invoice_amount", "amount", "total_amount", "gross_amount", "invoice_total"],
    "gstin": ["gstin", "gst_number", "gst_no", "vendor_gstin", "supplier_gstin"],
    "invoice_date": ["invoice_date", "date", "bill_date"],
    "po_amount": ["po_amount", "order_amount", "purchase_order_amount", "total_po_amount"],
    "grn_id": ["grn_id", "grn_number", "receipt_id", "goods_receipt_id"],
    "received_amount": ["received_amount", "receipt_amount", "goods_received_amount", "accepted_amount"],
    "receipt_date": ["receipt_date", "grn_date", "received_date"],
}

def normalize_columns(frame):
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    rename = {}
    for canonical, aliases in ALIASES.items():
        for col in frame.columns:
            if col in aliases:
                rename[col] = canonical
                break
    return frame.rename(columns=rename)

def read_csv(upload):
    try:
        return pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Could not read **{upload.name}** as CSV: {exc}")
        return None

def clean_text(value):
    return "" if pd.isna(value) else str(value).strip()

def clean_gstin(value):
    return re.sub(r"\s+", "", clean_text(value)).upper()

with st.sidebar:
    st.header("Configuration")
    tolerance = st.number_input("Amount tolerance (₹)", min_value=0.0, value=1.0, step=1.0)
    hold_threshold = st.number_input("Hold invoices above (₹)", min_value=0.0, value=1000000.0, step=50000.0, format="%.2f")
    st.caption("An amount difference within the tolerance is treated as a match.")
    st.divider()
    st.subheader("Optional Ollama")
    use_ollama = st.checkbox("Enable explanation / email draft with local Ollama", value=False)
    ollama_model = st.text_input("Ollama model", value="llama3.2:3b")
    st.caption("Ollama only explains calculated facts. It cannot decide invoice status.")

c1, c2, c3 = st.columns(3)
with c1:
    invoices_file = st.file_uploader("Upload invoices.csv", type=["csv"], key="invoices")
with c2:
    po_file = st.file_uploader("Upload purchase_orders.csv", type=["csv"], key="pos")
with c3:
    grn_file = st.file_uploader("Upload goods_receipts.csv", type=["csv"], key="grns")

if not all([invoices_file, po_file, grn_file]):
    st.info("Upload all three CSV files to begin. Common column aliases are accepted.")
    with st.expander("Required columns"):
        st.write("**invoices.csv:** invoice_id, vendor, po_id, invoice_amount, invoice_date; GSTIN is optional.")
        st.write("**purchase_orders.csv:** po_id, vendor, po_amount; GSTIN is optional.")
        st.write("**goods_receipts.csv:** grn_id, po_id, received_amount, receipt_date.")
    st.stop()

raw_invoices, raw_pos, raw_grns = read_csv(invoices_file), read_csv(po_file), read_csv(grn_file)
if any(x is None for x in [raw_invoices, raw_pos, raw_grns]):
    st.stop()
invoices, pos, grns = map(normalize_columns, [raw_invoices, raw_pos, raw_grns])

# GSTIN is optional; create a blank column when it is not provided.
for frame in [invoices, pos]:
    if "gstin" not in frame.columns:
        frame["gstin"] = ""

missing = []
for label, frame, required in [
    ("invoices.csv", invoices, ["invoice_id", "vendor", "po_id", "invoice_amount", "invoice_date"]),
    ("purchase_orders.csv", pos, ["po_id", "vendor", "po_amount"]),
    ("goods_receipts.csv", grns, ["grn_id", "po_id", "received_amount", "receipt_date"]),
]:
    absent = [c for c in required if c not in frame.columns]
    if absent:
        missing.append(f"**{label}** is missing: {', '.join(absent)}")
if missing:
    st.error("Please fix the following required-column issues:\n\n" + "\n\n".join(f"- {m}" for m in missing))
    st.stop()

for frame, amount_col in [(invoices, "invoice_amount"), (pos, "po_amount"), (grns, "received_amount")]:
    frame[amount_col] = pd.to_numeric(frame[amount_col], errors="coerce")
for frame, cols in [(invoices, ["invoice_id", "vendor", "po_id"]), (pos, ["vendor", "po_id"]), (grns, ["grn_id", "po_id"])]:
    for col in cols:
        frame[col] = frame[col].map(clean_text)
invoices["gstin"] = invoices["gstin"].map(clean_gstin)
pos["gstin"] = pos["gstin"].map(clean_gstin)
invoice_grn_col = next((c for c in ["grn_id", "grn_number", "receipt_id"] if c in invoices.columns), None)

# Matching, validation, calculations, and decisions are kept in the main workflow.
results = []
for _, inv in invoices.iterrows():
    invoice_id, vendor, po_id = clean_text(inv["invoice_id"]), clean_text(inv["vendor"]), clean_text(inv["po_id"])
    amount, gstin = inv["invoice_amount"], clean_gstin(inv.get("gstin", ""))
    reasons, hard_failures, needs_verification = [], [], []

    if not invoice_id: needs_verification.append("Invoice ID is missing.")
    if not vendor: needs_verification.append("Vendor is missing.")
    if not po_id: needs_verification.append("PO ID is missing.")
    if pd.isna(amount) or amount < 0: needs_verification.append("Invoice amount is missing, invalid, or negative.")

    duplicate_mask = ((invoices["vendor"].str.casefold() == vendor.casefold()) &
                      (invoices["po_id"].str.casefold() == po_id.casefold()) &
                      invoices["invoice_amount"].notna() &
                      (invoices["invoice_amount"].round(2) == (round(float(amount), 2) if pd.notna(amount) else -999999999)))
    duplicate_ids = invoices.loc[duplicate_mask, "invoice_id"].map(clean_text).tolist()
    is_duplicate = bool(invoice_id and len(duplicate_ids) > 1)
    if is_duplicate:
        hard_failures.append("Possible duplicate invoice: same vendor, PO, and amount appear more than once.")

    po_matches = pos[pos["po_id"].str.casefold() == po_id.casefold()] if po_id else pos.iloc[0:0]
    po_row = None
    if len(po_matches) == 0:
        needs_verification.append("No matching purchase order found.")
    else:
        vendor_matches = po_matches[po_matches["vendor"].str.casefold() == vendor.casefold()]
        if len(vendor_matches) == 0:
            hard_failures.append("PO vendor does not match invoice vendor.")
        else:
            po_row = vendor_matches.iloc[0]
            if len(vendor_matches) > 1:
                needs_verification.append("Multiple purchase orders match the same PO ID and vendor.")
            po_amount = po_row["po_amount"]
            if pd.isna(po_amount) or po_amount < 0:
                needs_verification.append("Matched PO amount is missing or invalid.")
            elif pd.notna(amount) and amount > po_amount + tolerance:
                hard_failures.append(f"Invoice amount exceeds PO amount by ₹{amount - po_amount:,.2f}, beyond tolerance.")
            elif pd.notna(amount) and abs(amount - po_amount) > tolerance:
                needs_verification.append(f"Invoice and PO amounts differ by ₹{abs(amount - po_amount):,.2f}.")
            po_gstin = clean_gstin(po_row.get("gstin", ""))
            if gstin and po_gstin and gstin != po_gstin:
                hard_failures.append("Invoice GSTIN does not match the PO GSTIN.")
            elif not gstin or not po_gstin:
                needs_verification.append("GSTIN is missing on the invoice or PO; consistency could not be confirmed.")

    grn_matches = grns[grns["po_id"].str.casefold() == po_id.casefold()] if po_id else grns.iloc[0:0]
    invoice_grn = clean_text(inv.get(invoice_grn_col, "")) if invoice_grn_col else ""
    if invoice_grn:
        exact_grn = grn_matches[grn_matches["grn_id"].str.casefold() == invoice_grn.casefold()]
        if len(exact_grn):
            grn_matches = exact_grn
        else:
            needs_verification.append(f"Invoice GRN reference '{invoice_grn}' was not found for this PO.")
    if len(grn_matches) == 0:
        needs_verification.append("No goods receipt found for this PO.")
        received_total, matched_grns = None, ""
    else:
        received_total = grn_matches["received_amount"].sum(min_count=1)
        matched_grns = ", ".join(grn_matches["grn_id"].map(clean_text).tolist())
        if pd.isna(received_total) or received_total < 0:
            needs_verification.append("Goods receipt amount is missing or invalid.")
        elif pd.notna(amount) and amount > received_total + tolerance:
            hard_failures.append(f"Invoice amount exceeds total goods receipt amount by ₹{amount - received_total:,.2f}, beyond tolerance.")
        elif pd.notna(amount) and abs(amount - received_total) > tolerance:
            needs_verification.append(f"Invoice and goods receipt amounts differ by ₹{abs(amount - received_total):,.2f}.")

    if pd.notna(amount) and amount > hold_threshold:
        needs_verification.append(f"Invoice amount exceeds the configured hold threshold of ₹{hold_threshold:,.2f}.")
    if gstin and not re.fullmatch(r"\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]", gstin):
        needs_verification.append("Invoice GSTIN format looks invalid; this is not government verification.")

    if hard_failures:
        status, reasons = "REJECT", hard_failures + needs_verification
    elif needs_verification:
        status, reasons = "HOLD", needs_verification
    else:
        status, reasons = "APPROVE", ["Invoice, PO, GRN, vendor, amount, and available GSTIN checks passed."]

    results.append({
        "invoice_id": invoice_id, "vendor": vendor, "po_id": po_id, "invoice_amount": amount,
        "invoice_date": clean_text(inv.get("invoice_date", "")), "invoice_gstin": gstin,
        "po_amount": po_row["po_amount"] if po_row is not None else None,
        "po_gstin": clean_gstin(po_row.get("gstin", "")) if po_row is not None else "",
        "received_amount": received_total, "matched_grns": matched_grns,
        "duplicate": "Yes" if is_duplicate else "No", "status": status,
        "decision_reasons": " ".join(dict.fromkeys(reasons)), "original_status": status, "override_reason": ""
    })
result_df = pd.DataFrame(results)

st.subheader("Manual review and override")
invoice_options = result_df["invoice_id"].astype(str).tolist()
selected_invoice = st.selectbox("Select invoice to inspect", invoice_options)
selected_index = result_df.index[result_df["invoice_id"].astype(str) == str(selected_invoice)][0]
selected = result_df.loc[selected_index]
a, b, c, d = st.columns(4)
a.metric("Invoice amount", "—" if pd.isna(selected["invoice_amount"]) else f"₹{selected['invoice_amount']:,.2f}")
b.metric("PO amount", "—" if pd.isna(selected["po_amount"]) else f"₹{selected['po_amount']:,.2f}")
c.metric("Goods received", "—" if pd.isna(selected["received_amount"]) else f"₹{selected['received_amount']:,.2f}")
d.metric("Rule-based status", selected["status"])
st.write(f"**Vendor:** {selected['vendor'] or 'Not supplied'} | **PO:** {selected['po_id'] or 'Not supplied'}")
st.write(f"**Reasons:** {selected['decision_reasons']}")
st.caption("GST checks compare supplied values and inspect a common GSTIN pattern. They do not verify registration with a government system.")

with st.form("override_form"):
    override_status = st.selectbox("Manual override status", ["No change", "APPROVE", "HOLD", "REJECT"])
    override_reason = st.text_input("Override reason (required when changing status)")
    submit_override = st.form_submit_button("Apply override")
    if submit_override:
        if override_status != "No change" and not override_reason.strip():
            st.error("Enter a reason before applying a manual override.")
        elif override_status != "No change":
            result_df.loc[selected_index, "status"] = override_status
            result_df.loc[selected_index, "override_reason"] = override_reason.strip()
            result_df.loc[selected_index, "decision_reasons"] += f" | MANUAL OVERRIDE: {override_status} — {override_reason.strip()}"
            st.success(f"Manual override applied: {override_status}")
        else:
            st.info("No override applied.")

st.divider()
st.subheader("Dashboard")
count, approved = len(result_df), int((result_df["status"] == "APPROVE").sum())
hold, rejected = int((result_df["status"] == "HOLD").sum()), int((result_df["status"] == "REJECT").sum())
total_amount = result_df["invoice_amount"].sum()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Invoices", count); k2.metric("Approved", approved); k3.metric("On hold", hold); k4.metric("Rejected", rejected)
k5.metric("Invoice value", f"₹{total_amount:,.2f}" if pd.notna(total_amount) else "—")
chart1, chart2 = st.columns(2)
with chart1:
    counts = result_df["status"].value_counts().rename_axis("Status").reset_index(name="Invoices")
    st.plotly_chart(px.bar(counts, x="Status", y="Invoices", title="Invoice count by status"), use_container_width=True)
with chart2:
    amounts = result_df.groupby("status", dropna=False)["invoice_amount"].sum(min_count=1).reset_index()
    st.plotly_chart(px.bar(amounts, x="status", y="invoice_amount", title="Invoice amount by status", labels={"status": "Status", "invoice_amount": "Amount (₹)"}), use_container_width=True)

st.subheader("Filter and review invoices")
f1, f2 = st.columns(2)
with f1:
    statuses = st.multiselect("Filter by status", ["APPROVE", "HOLD", "REJECT"], default=["APPROVE", "HOLD", "REJECT"])
with f2:
    vendors = sorted([v for v in result_df["vendor"].dropna().unique().tolist() if str(v).strip()])
    selected_vendors = st.multiselect("Filter by vendor", vendors, default=vendors)
filtered = result_df[result_df["status"].isin(statuses)]
if selected_vendors:
    filtered = filtered[filtered["vendor"].isin(selected_vendors)]
elif vendors:
    filtered = filtered.iloc[0:0]
columns = ["invoice_id", "vendor", "po_id", "invoice_amount", "po_amount", "received_amount", "duplicate", "status", "decision_reasons", "override_reason"]
st.dataframe(filtered[columns], use_container_width=True, hide_index=True)
st.download_button("Download full results as CSV", data=result_df.to_csv(index=False).encode("utf-8-sig"), file_name="invoice_matching_results.csv", mime="text/csv")

if use_ollama:
    st.subheader("Optional Ollama explanation / email draft")
    if st.button("Generate explanation for selected invoice"):
        facts = (f"Invoice ID: {selected['invoice_id']}; Vendor: {selected['vendor']}; PO: {selected['po_id']}; "
                 f"Invoice amount: {selected['invoice_amount']}; PO amount: {selected['po_amount']}; "
                 f"Received amount: {selected['received_amount']}; Rule-based status: {selected['status']}; "
                 f"Reasons: {selected['decision_reasons']}; Override reason: {selected['override_reason']}")
        try:
            import ollama
            answer = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": (
                "Write a concise finance-team explanation and optionally a draft email using only the facts below. "
                "Do not invent facts or change the rule-based status. State that deterministic rules set the status.\n\n" + facts
            )}])
            st.write(answer["message"]["content"])
        except Exception as exc:
            st.warning("Ollama is unavailable or the model is not installed. The dashboard remains functional. " + str(exc))

st.caption("Decision policy: REJECT for configured hard failures such as duplicate or vendor mismatch; HOLD for missing/uncertain information or amounts above the threshold; APPROVE only when all implemented checks pass.")
