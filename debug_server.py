#!/usr/bin/env python3
"""Debug script to test server components."""

import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Simple test models
class TestRequest(BaseModel):
    content: str
    doc_id: str

class TestResponse(BaseModel):
    success: bool
    doc_id: str
    message: str

# Create simple FastAPI app
app = FastAPI(title="Debug Server")

# Simple global state
debug_state = {
    "test_value": "initialized",
    "router": None,
    "registry": None
}

@app.get("/debug/state")
async def debug_state_endpoint():
    """Debug endpoint to check state."""
    return {
        "debug_state": debug_state,
        "router_info": {
            "exists": "router" in debug_state,
            "value": str(debug_state.get("router", "NOT_FOUND")),
            "type": str(type(debug_state.get("router", "NOT_FOUND"))),
            "is_none": debug_state.get("router") is None
        }
    }

@app.post("/debug/test-analysis", response_model=TestResponse)
async def test_analysis(request: TestRequest):
    """Test analysis endpoint."""
    try:
        # Test the same logic as the main server
        if debug_state.get("router") is not None:
            return TestResponse(
                success=False,
                doc_id=request.doc_id,
                message="Router is not None - this should not happen"
            )
        else:
            return TestResponse(
                success=True,
                doc_id=request.doc_id,
                message="Router is None - using fallback analysis"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "message": "Debug server is running"}

if __name__ == "__main__":
    print("Starting debug server...")
    uvicorn.run(app, host="0.0.0.0", port=8002) 