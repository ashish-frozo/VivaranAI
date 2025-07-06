"""
Test data for end-to-end testing of MedBillGuardAgent system.
Contains various medical bill scenarios to test different edge cases and workflows.
"""

import base64
from typing import Dict, Any

# Sample Medical Bill 1: High overcharge scenario with duplicates
SAMPLE_BILL_1_TEXT = """
APOLLO HOSPITALS
Delhi Branch
Medical Bill Summary

Patient: John Doe
Date: 2024-01-15
Bill No: APL/2024/001234

CONSULTATION CHARGES:
- Doctor Consultation (General Medicine): ₹2,500
- Doctor Consultation (Cardiology): ₹3,000  
- Doctor Consultation (General Medicine): ₹2,500
- Emergency Consultation: ₹1,500

DIAGNOSTIC TESTS:
- Complete Blood Count (CBC): ₹800
- Complete Blood Count (CBC): ₹800
- Complete Blood Count (CBC): ₹800
- Complete Blood Count (CBC): ₹800
- Chest X-Ray: ₹1,200
- ECG: ₹600
- Lipid Profile: ₹1,500
- Liver Function Test: ₹1,200

ROOM CHARGES:
- General Ward (2 days): ₹4,000
- AC Charges: ₹1,000

PHARMACY:
- Paracetamol 500mg (10 tablets): ₹150
- Aspirin 75mg (30 tablets): ₹200
- Metformin 500mg (30 tablets): ₹300

MISCELLANEOUS:
- Registration Fee: ₹500
- Medical Records Fee: ₹200
- Nursing Charges: ₹2,000

TOTAL BILL AMOUNT: ₹23,550

Payment Mode: Cash
Discharge Date: 2024-01-17
"""

# Sample Medical Bill 2: Normal bill with minimal issues
SAMPLE_BILL_2_TEXT = """
MAX HEALTHCARE
Saket, New Delhi
Patient Bill

Patient: Sarah Singh
Date: 2024-02-20
Bill No: MAX/2024/005678

CONSULTATION CHARGES:
- Doctor Consultation (Pediatrics): ₹800

DIAGNOSTIC TESTS:
- Complete Blood Count (CBC): ₹300
- Urine Routine: ₹200

PHARMACY:
- Paracetamol Syrup: ₹80
- Vitamin D3 Drops: ₹120

TOTAL BILL AMOUNT: ₹1,500

Payment Mode: UPI
Discharge Date: 2024-02-20
"""

# Sample Medical Bill 3: Complex bill with prohibited items
SAMPLE_BILL_3_TEXT = """
FORTIS HOSPITAL
Gurgaon
Detailed Medical Bill

Patient: Rajesh Kumar
Date: 2024-03-10
Bill No: FOR/2024/009876

CONSULTATION CHARGES:
- Doctor Consultation (Orthopedics): ₹1,800
- Specialist Consultation (Anesthesia): ₹2,200

SURGICAL PROCEDURES:
- Knee Replacement Surgery: ₹150,000
- OT Charges: ₹25,000
- Anesthesia Charges: ₹15,000

DIAGNOSTIC TESTS:
- Pre-operative Blood Tests: ₹2,500
- X-Ray Knee (AP/Lateral): ₹800
- MRI Knee: ₹8,500

ROOM CHARGES:
- Private Room (3 days): ₹15,000
- ICU Charges (1 day): ₹12,000

PHARMACY:
- Pain Medication: ₹800
- Antibiotics: ₹1,200
- Surgical Dressing Kit: ₹500

MISCELLANEOUS:
- Registration Fee: ₹500
- Medical Records Fee: ₹300
- Physiotherapy Session: ₹1,500
- Ambulance Charges: ₹2,000
- Food Charges: ₹1,800
- TV/WiFi Charges: ₹600
- Visitor Pass Charges: ₹200

TOTAL BILL AMOUNT: ₹238,500

Payment Mode: Insurance + Cash
Discharge Date: 2024-03-13
"""

# Sample Medical Bill 4: Hindi text bill (multilingual testing)
SAMPLE_BILL_4_TEXT = """
सरकारी अस्पताल
दिल्ली
मेडिकल बिल

मरीज़ का नाम: अमित शर्मा
दिनांक: 2024-01-25
बिल नंबर: GOVT/2024/001122

परामर्श शुल्क:
- डॉक्टर परामर्श (सामान्य चिकित्सा): ₹500

जाँच शुल्क:
- रक्त जाँच (CBC): ₹150
- छाती का एक्स-रे: ₹100

दवाई:
- पेरासिटामोल टैबलेट: ₹30

कुल राशि: ₹780

भुगतान मोड: नकद
छुट्टी की तारीख: 2024-01-25
"""

# Sample Medical Bill 5: Emergency bill with time-sensitive charges
SAMPLE_BILL_5_TEXT = """
AIIMS TRAUMA CENTER
Emergency Department
New Delhi

Patient: Emergency Case #2024030801
Date: 2024-03-08
Time: 02:30 AM
Bill No: AIIMS/EMERGENCY/2024/001

EMERGENCY CHARGES:
- Emergency Room Charges (Night): ₹2,000
- Doctor Consultation (Emergency): ₹1,500
- Trauma Consultation: ₹2,500

DIAGNOSTIC TESTS:
- CT Scan Head: ₹4,500
- Chest X-Ray: ₹400
- Blood Tests (Emergency): ₹1,200

PROCEDURES:
- Wound Suturing: ₹800
- IV Fluids: ₹600
- Injection Administration: ₹300

PHARMACY:
- Emergency Medications: ₹500
- Pain Injections: ₹400

MISCELLANEOUS:
- Ambulance Charges: ₹1,500
- Emergency Bed Charges (4 hours): ₹2,000

TOTAL BILL AMOUNT: ₹16,700

Payment Mode: Emergency Fund
Discharge Date: 2024-03-08
Time: 08:15 AM
"""

def encode_text_as_base64(text: str) -> str:
    """Convert text to base64 format for API testing."""
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

# Test scenarios with expected outcomes
TEST_SCENARIOS = [
    {
        "name": "High Overcharge with Duplicates",
        "description": "Apollo Hospitals bill with multiple duplicate charges and potential overcharging",
        "bill_text": SAMPLE_BILL_1_TEXT,
        "expected_findings": {
            "duplicates_found": True,
            "duplicate_count": 7,  # 3 duplicate CBC tests + 2 duplicate consultations
            "overcharge_detected": True,
            "confidence_level": "high",
            "estimated_overcharge": 5000,  # Approximate
            "red_flags": ["duplicate_charges", "potential_overcharge", "excessive_billing"]
        }
    },
    {
        "name": "Normal Bill - Minimal Issues",
        "description": "MAX Healthcare pediatric consultation with standard charges",
        "bill_text": SAMPLE_BILL_2_TEXT,
        "expected_findings": {
            "duplicates_found": False,
            "duplicate_count": 0,
            "overcharge_detected": False,
            "confidence_level": "high",
            "estimated_overcharge": 0,
            "red_flags": []
        }
    },
    {
        "name": "Complex Surgery with Prohibited Items",
        "description": "Fortis Hospital complex surgery bill with potential prohibited charges",
        "bill_text": SAMPLE_BILL_3_TEXT,
        "expected_findings": {
            "duplicates_found": False,
            "duplicate_count": 0,
            "overcharge_detected": True,
            "confidence_level": "medium",
            "prohibited_items": ["food_charges", "tv_wifi_charges", "visitor_pass_charges"],
            "red_flags": ["prohibited_charges", "non_medical_charges"]
        }
    },
    {
        "name": "Hindi Language Bill",
        "description": "Government hospital bill in Hindi to test multilingual OCR",
        "bill_text": SAMPLE_BILL_4_TEXT,
        "expected_findings": {
            "duplicates_found": False,
            "duplicate_count": 0,
            "overcharge_detected": False,
            "confidence_level": "medium",  # May be lower due to language processing
            "language_detected": "hindi",
            "red_flags": []
        }
    },
    {
        "name": "Emergency Department Bill",
        "description": "AIIMS emergency department bill with time-sensitive charges",
        "bill_text": SAMPLE_BILL_5_TEXT,
        "expected_findings": {
            "duplicates_found": False,
            "duplicate_count": 0,
            "overcharge_detected": False,
            "confidence_level": "high",
            "emergency_charges": True,
            "red_flags": []
        }
    }
]

def get_test_scenario(scenario_name: str) -> Dict[str, Any]:
    """Get a specific test scenario by name."""
    for scenario in TEST_SCENARIOS:
        if scenario["name"] == scenario_name:
            return scenario
    raise ValueError(f"Test scenario '{scenario_name}' not found")

def get_all_test_scenarios() -> list:
    """Get all available test scenarios."""
    return TEST_SCENARIOS

def prepare_api_payload(scenario_name: str) -> Dict[str, Any]:
    """Prepare API payload for a test scenario."""
    scenario = get_test_scenario(scenario_name)
    return {
        "file_content": encode_text_as_base64(scenario["bill_text"]),
        "filename": f"test_bill_{scenario_name.lower().replace(' ', '_')}.txt",
        "file_type": "text/plain"
    } 