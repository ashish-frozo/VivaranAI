"""
Explanation Builder for MedBillGuardAgent
Generates Markdown and SSML explanations for analysis results.
"""
from typing import List, Dict, Any, Tuple

def build_explanation(verdict: str, red_flags: List[Dict[str, Any]], total_overcharge: float, total_bill_amount: float) -> Tuple[str, str]:
    """
    Generate Markdown and SSML explanations for the analysis result.
    Returns (markdown, ssml)
    """
    if verdict == "ok":
        md = (
            "**Result:** ✅ No over-charges detected. Your bill appears reasonable.\n\n"
            "- **Total Bill Amount:** ₹{:.2f}\n"
            "- **Total Overcharge:** ₹{:.2f}\n"
            "- **Red Flags:** None\n"
            "\nIf you have questions, you can request a detailed breakdown."
        ).format(total_bill_amount, total_overcharge)
        ssml = (
            "<speak>"
            "No over-charges detected. Your bill appears reasonable. "
            "Total bill amount is ₹{:.2f}. "
            "No red flags were found."
            "</speak>"
        ).format(total_bill_amount)
        return md, ssml

    # If there are red flags
    md = (
        f"**Result:** {'⚠️' if verdict == 'warning' else '🚨'} {verdict.title()} detected.\n\n"
        f"- **Total Bill Amount:** ₹{total_bill_amount:.2f}\n"
        f"- **Total Overcharge:** ₹{total_overcharge:.2f}\n"
        f"- **Red Flags:** {len(red_flags)} found\n\n"
    )
    for i, flag in enumerate(red_flags, 1):
        desc = flag.get("description") or flag.get("type") or "Issue"
        amt = flag.get("overcharge_amount", 0)
        sev = flag.get("severity", "warning")
        md += f"{i}. **{desc}** (Severity: {sev})"
        if amt:
            md += f" — Overcharge: ₹{amt:.2f}"
        md += "\n"
    md += "\nPlease review the flagged items. If you need help, contact your hospital or insurance ombudsman."

    # SSML version
    ssml = (
        f"<speak>"
        f"{len(red_flags)} potential issue{'s' if len(red_flags) != 1 else ''} detected. "
        f"Total bill amount is ₹{total_bill_amount:.2f}. "
        f"Total overcharge is ₹{total_overcharge:.2f}. "
    )
    for flag in red_flags:
        desc = flag.get("description") or flag.get("type") or "Issue"
        amt = flag.get("overcharge_amount", 0)
        sev = flag.get("severity", "warning")
        ssml += f" {desc}. Severity: {sev}."
        if amt:
            ssml += f" Overcharge: ₹{amt:.2f}."
    ssml += " Please review the flagged items. </speak>"
    return md, ssml 