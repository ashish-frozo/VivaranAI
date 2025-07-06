"""Tests for Reference Data Loader."""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from medbillguardagent.reference_data_loader import (
    DrugRate,
    ProhibitedItem,
    ReferenceDataLoader,
    ReferenceRate,
)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_cghs_data():
    """Sample CGHS rates data."""
    return {
        "rates": [
            {
                "code": "CGHS-001",
                "name": "Consultation - General Medicine",
                "category": "consultation",
                "rate": 500.0,
                "source": "CGHS",
                "state_code": None,
                "effective_date": "2023-01-01T00:00:00Z"
            },
            {
                "code": "CGHS-002", 
                "name": "Blood Test - Complete Blood Count",
                "category": "diagnostic",
                "rate": 200.0,
                "source": "CGHS",
                "state_code": None,
                "effective_date": "2023-01-01T00:00:00Z"
            }
        ]
    }


@pytest.fixture
def sample_esi_data():
    """Sample ESI rates data."""
    return {
        "rates": [
            {
                "code": "ESI-001",
                "name": "Consultation - Specialist",
                "category": "consultation", 
                "rate": 800.0,
                "source": "ESI",
                "state_code": None,
                "effective_date": "2023-01-01T00:00:00Z"
            }
        ]
    }


@pytest.fixture
def sample_nppa_data():
    """Sample NPPA drug data."""
    return {
        "drugs": [
            {
                "drug_name": "Paracetamol",
                "brand_name": "Crocin",
                "strength": "500mg",
                "dosage_form": "Tablet",
                "pack_size": "10 tablets",
                "mrp": 25.0,
                "manufacturer": "GSK",
                "updated_date": "2023-01-01T00:00:00Z",
                "category": "Analgesic",
                "schedule": "OTC"
            },
            {
                "drug_name": "Amoxicillin",
                "brand_name": None,
                "strength": "250mg",
                "dosage_form": "Capsule",
                "pack_size": "10 capsules",
                "mrp": 45.0,
                "manufacturer": "Cipla",
                "updated_date": "2023-01-01T00:00:00Z",
                "category": "Antibiotic",
                "schedule": "H"
            }
        ]
    }


@pytest.fixture
def sample_nppa_csv_data():
    """Sample NPPA drug CSV data."""
    return """drug_name,brand_name,strength,dosage_form,pack_size,mrp,manufacturer,updated_date,category,schedule
Paracetamol,Crocin,500mg,Tablet,10 tablets,25.0,GSK,2023-01-01T00:00:00Z,Analgesic,OTC
Paracetamol,Dolo,650mg,Tablet,15 tablets,35.0,Micro Labs,2023-01-01T00:00:00Z,Analgesic,OTC
Amoxicillin,Amoxil,250mg,Capsule,10 capsules,45.0,Cipla,2023-01-01T00:00:00Z,Antibiotic,H
Omeprazole,Omez,20mg,Capsule,10 capsules,35.0,Dr. Reddy's,2023-01-01T00:00:00Z,Antacid,H"""


@pytest.fixture
def sample_prohibited_data():
    """Sample prohibited items data."""
    return {
        "prohibited_items": [
            {
                "name": "Cosmetic Surgery",
                "category": "surgery",
                "reason": "Not covered under insurance",
                "source": "CGHS Guidelines"
            },
            {
                "name": "Hair Transplant",
                "category": "cosmetic",
                "reason": "Elective procedure",
                "source": "ESI Rules"
            }
        ]
    }


@pytest.fixture
async def loader_with_data(temp_data_dir, sample_cghs_data, sample_esi_data, sample_nppa_data, sample_nppa_csv_data, sample_prohibited_data):
    """Create a loader with sample data files."""
    
    # Create data files
    cghs_file = temp_data_dir / "cghs_rates_2023.json"
    esi_file = temp_data_dir / "esi_rates.json"
    nppa_file = temp_data_dir / "nppa_mrp.json"
    nppa_csv_file = temp_data_dir / "nppa_mrp.csv"
    prohibited_file = temp_data_dir / "prohibited.json"
    
    # Write sample data
    with open(cghs_file, 'w') as f:
        json.dump(sample_cghs_data, f)
    with open(esi_file, 'w') as f:
        json.dump(sample_esi_data, f)
    with open(nppa_file, 'w') as f:
        json.dump(sample_nppa_data, f)
    with open(nppa_csv_file, 'w') as f:
        f.write(sample_nppa_csv_data)
    with open(prohibited_file, 'w') as f:
        json.dump(sample_prohibited_data, f)
    
    # Create state tariffs directory with sample data
    state_dir = temp_data_dir / "state_tariffs"
    state_dir.mkdir()
    
    state_data = {
        "rates": [
            {
                "code": "DL-001",
                "name": "Delhi State Rate - Consultation",
                "category": "consultation",
                "rate": 600.0,
                "source": "DELHI",
                "state_code": "DL",
                "effective_date": "2023-01-01T00:00:00Z"
            }
        ]
    }
    
    with open(state_dir / "dl.json", 'w') as f:
        json.dump(state_data, f)
    
    # Initialize loader
    loader = ReferenceDataLoader(data_dir=temp_data_dir)
    await loader.initialize()
    
    return loader


class TestReferenceDataLoader:
    """Test cases for ReferenceDataLoader."""

    def test_initialization(self, temp_data_dir):
        """Test loader initialization."""
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        
        assert loader.data_dir == temp_data_dir
        assert len(loader._cghs_rates) == 0
        assert len(loader._esi_rates) == 0
        assert len(loader._nppa_drugs) == 0

    @pytest.mark.asyncio
    async def test_load_cghs_rates(self, temp_data_dir, sample_cghs_data):
        """Test loading CGHS rates."""
        # Create CGHS data file
        cghs_file = temp_data_dir / "cghs_rates_2023.json"
        with open(cghs_file, 'w') as f:
            json.dump(sample_cghs_data, f)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        rates = await loader.load_cghs_rates()
        
        assert len(rates) == 2
        assert "CGHS-001" in rates
        assert rates["CGHS-001"].name == "Consultation - General Medicine"
        assert rates["CGHS-001"].rate == 500.0

    @pytest.mark.asyncio
    async def test_load_cghs_rates_file_not_found(self, temp_data_dir):
        """Test loading CGHS rates when file doesn't exist."""
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        rates = await loader.load_cghs_rates()
        
        assert len(rates) == 0

    @pytest.mark.asyncio
    async def test_load_esi_rates(self, temp_data_dir, sample_esi_data):
        """Test loading ESI rates."""
        # Create ESI data file
        esi_file = temp_data_dir / "esi_rates.json"
        with open(esi_file, 'w') as f:
            json.dump(sample_esi_data, f)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        rates = await loader.load_esi_rates()
        
        assert len(rates) == 1
        assert "ESI-001" in rates
        assert rates["ESI-001"].name == "Consultation - Specialist"
        assert rates["ESI-001"].rate == 800.0

    @pytest.mark.asyncio
    async def test_load_nppa_drugs(self, temp_data_dir, sample_nppa_data):
        """Test loading NPPA drug data from JSON."""
        # Create NPPA data file
        nppa_file = temp_data_dir / "nppa_mrp.json"
        with open(nppa_file, 'w') as f:
            json.dump(sample_nppa_data, f)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        drugs = await loader.load_nppa_drugs()
        
        assert len(drugs) >= 2
        assert "paracetamol" in drugs
        assert drugs["paracetamol"].mrp == 25.0
        assert drugs["paracetamol"].brand_name == "Crocin"
        assert drugs["paracetamol"].category == "Analgesic"
        assert drugs["paracetamol"].schedule == "OTC"

    @pytest.mark.asyncio
    async def test_load_nppa_drugs_csv(self, temp_data_dir, sample_nppa_csv_data):
        """Test loading NPPA drug data from CSV."""
        # Create NPPA CSV file
        nppa_csv_file = temp_data_dir / "nppa_mrp.csv"
        with open(nppa_csv_file, 'w') as f:
            f.write(sample_nppa_csv_data)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        drugs = await loader.load_nppa_drugs()
        
        assert len(drugs) >= 4
        
        # Test exact drug name match (last loaded entry wins)
        assert "paracetamol" in drugs
        # Note: "Dolo" is the last Paracetamol entry in CSV, so it overwrites "Crocin"
        assert drugs["paracetamol"].brand_name == "Dolo"
        assert drugs["paracetamol"].mrp == 35.0
        
        # Test strength-specific key
        assert "paracetamol_500mg" in drugs
        assert drugs["paracetamol_500mg"].strength == "500mg"
        
        # Test different strength
        assert "paracetamol_650mg" in drugs
        assert drugs["paracetamol_650mg"].brand_name == "Dolo"
        assert drugs["paracetamol_650mg"].mrp == 35.0

    @pytest.mark.asyncio
    async def test_enhanced_drug_search(self, temp_data_dir, sample_nppa_csv_data):
        """Test enhanced drug search with strength and brand name."""
        # Create NPPA CSV file
        nppa_csv_file = temp_data_dir / "nppa_mrp.csv"
        with open(nppa_csv_file, 'w') as f:
            f.write(sample_nppa_csv_data)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        await loader.load_nppa_drugs()
        
        # Test search with strength
        drug = await loader.find_drug_rate("paracetamol", strength="500mg")
        assert drug is not None
        assert drug.strength == "500mg"
        assert drug.brand_name == "Crocin"
        
        # Test search with different strength
        drug = await loader.find_drug_rate("paracetamol", strength="650mg")
        assert drug is not None
        assert drug.strength == "650mg"
        assert drug.brand_name == "Dolo"
        
        # Test search with brand name
        drug = await loader.find_drug_rate("paracetamol", brand_name="Dolo")
        assert drug is not None
        assert drug.brand_name == "Dolo"
        
        # Test search with both strength and brand
        drug = await loader.find_drug_rate("paracetamol", strength="500mg", brand_name="Crocin")
        assert drug is not None
        assert drug.strength == "500mg"
        assert drug.brand_name == "Crocin"

    @pytest.mark.asyncio
    async def test_load_state_rates(self, temp_data_dir):
        """Test loading state-specific rates."""
        # Create state tariffs directory
        state_dir = temp_data_dir / "state_tariffs"
        state_dir.mkdir()
        
        state_data = {
            "rates": [
                {
                    "code": "DL-001",
                    "name": "Delhi Rate",
                    "category": "consultation",
                    "rate": 600.0,
                    "source": "DELHI",
                    "state_code": "DL",
                    "effective_date": "2023-01-01T00:00:00Z"
                }
            ]
        }
        
        with open(state_dir / "dl.json", 'w') as f:
            json.dump(state_data, f)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        state_rates = await loader.load_state_rates()
        
        assert len(state_rates) == 1
        assert "DL" in state_rates
        assert len(state_rates["DL"]) == 1
        assert "DL-001" in state_rates["DL"]

    @pytest.mark.asyncio
    async def test_load_prohibited_items(self, temp_data_dir, sample_prohibited_data):
        """Test loading prohibited items."""
        # Create prohibited items file
        prohibited_file = temp_data_dir / "prohibited.json"
        with open(prohibited_file, 'w') as f:
            json.dump(sample_prohibited_data, f)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        items = await loader.load_prohibited_items()
        
        assert len(items) == 2
        assert "cosmetic surgery" in items
        assert items["cosmetic surgery"].reason == "Not covered under insurance"

    @pytest.mark.asyncio
    async def test_find_procedure_rate(self, loader_with_data):
        """Test finding procedure rates."""
        loader = loader_with_data
        
        # Test exact match
        rate = await loader.find_procedure_rate("Consultation - General Medicine")
        assert rate is not None
        assert rate.code == "CGHS-001"
        assert rate.rate == 500.0
        
        # Test partial match
        rate = await loader.find_procedure_rate("blood test")
        assert rate is not None
        assert rate.code == "CGHS-002"
        
        # Test not found
        rate = await loader.find_procedure_rate("Non-existent procedure")
        assert rate is None

    @pytest.mark.asyncio
    async def test_find_procedure_rate_with_state(self, loader_with_data):
        """Test finding procedure rates with state preference."""
        loader = loader_with_data
        
        # Test state-specific rate
        rate = await loader.find_procedure_rate("consultation", state_code="DL")
        assert rate is not None
        assert rate.code == "DL-001"
        assert rate.rate == 600.0

    @pytest.mark.asyncio
    async def test_find_drug_rate(self, loader_with_data):
        """Test finding drug rates."""
        loader = loader_with_data
        
        # Test exact match
        drug = await loader.find_drug_rate("paracetamol")
        assert drug is not None
        assert drug.drug_name == "Paracetamol"
        assert drug.mrp == 35.0
        
        # Test partial match
        drug = await loader.find_drug_rate("amox")
        assert drug is not None
        assert drug.drug_name == "Amoxicillin"
        
        # Test not found
        drug = await loader.find_drug_rate("non-existent drug")
        assert drug is None

    @pytest.mark.asyncio
    async def test_is_prohibited_item(self, loader_with_data):
        """Test checking prohibited items."""
        loader = loader_with_data
        
        # Test exact match
        item = await loader.is_prohibited_item("cosmetic surgery")
        assert item is not None
        assert item.name == "Cosmetic Surgery"
        assert item.reason == "Not covered under insurance"
        
        # Test partial match
        item = await loader.is_prohibited_item("hair")
        assert item is not None
        assert item.name == "Hair Transplant"
        
        # Test not found
        item = await loader.is_prohibited_item("allowed procedure")
        assert item is None

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, loader_with_data):
        """Test getting cache statistics."""
        loader = loader_with_data
        
        stats = await loader.get_cache_stats()
        
        assert "data_sources" in stats
        assert stats["data_sources"]["cghs_rates"] == 2
        assert stats["data_sources"]["esi_rates"] == 1
        assert stats["data_sources"]["nppa_drugs"] == 7
        assert stats["data_sources"]["prohibited_items"] == 2
        
        assert "last_loaded" in stats
        assert "cghs" in stats["last_loaded"]

    @pytest.mark.asyncio
    async def test_refresh_data(self, loader_with_data):
        """Test refreshing data."""
        loader = loader_with_data
        
        # Should not raise any errors
        await loader.refresh_data(force=True)
        
        # Data should still be loaded
        assert len(loader._cghs_rates) == 2
        assert len(loader._esi_rates) == 1

    @pytest.mark.asyncio
    async def test_error_handling_invalid_json(self, temp_data_dir):
        """Test error handling with invalid JSON."""
        # Create invalid JSON file
        cghs_file = temp_data_dir / "cghs_rates_2023.json"
        with open(cghs_file, 'w') as f:
            f.write("invalid json content")
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        rates = await loader.load_cghs_rates()
        
        # Should return empty dict on error
        assert len(rates) == 0

    @pytest.mark.asyncio
    async def test_error_handling_missing_fields(self, temp_data_dir):
        """Test error handling with missing required fields."""
        # Create data with missing fields
        invalid_data = {
            "rates": [
                {
                    "code": "CGHS-001",
                    # Missing required fields
                }
            ]
        }
        
        cghs_file = temp_data_dir / "cghs_rates_2023.json"
        with open(cghs_file, 'w') as f:
            json.dump(invalid_data, f)
        
        loader = ReferenceDataLoader(data_dir=temp_data_dir)
        rates = await loader.load_cghs_rates()
        
        # Should return empty dict on validation error
        assert len(rates) == 0

    @pytest.mark.asyncio
    async def test_empty_search_terms(self, loader_with_data):
        """Test handling of empty search terms."""
        loader = loader_with_data
        
        # Test empty strings
        assert await loader.find_procedure_rate("") is None
        assert await loader.find_procedure_rate(None) is None
        assert await loader.find_drug_rate("") is None
        assert await loader.find_drug_rate(None) is None
        assert await loader.is_prohibited_item("") is None
        assert await loader.is_prohibited_item(None) is None

    @pytest.mark.asyncio
    async def test_case_insensitive_search(self, loader_with_data):
        """Test case-insensitive searching."""
        loader = loader_with_data
        
        # Test different cases
        rate = await loader.find_procedure_rate("CONSULTATION")
        assert rate is not None
        
        rate = await loader.find_procedure_rate("consultation")
        assert rate is not None
        
        drug = await loader.find_drug_rate("PARACETAMOL")
        assert drug is not None
        
        drug = await loader.find_drug_rate("ParaCetamol")
        assert drug is not None 