#!/usr/bin/env python3
"""
Production HTTP server for MedBillGuard frontend dashboard.
Serves the production HTML dashboard with CORS headers enabled.
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path
import webbrowser
from urllib.parse import urlparse

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers."""
    
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()
    
    def do_OPTIONS(self):
        """Handle preflight OPTIONS requests."""
        self.send_response(200)
        self.end_headers()
    
    def do_GET(self):
        """Handle GET requests, serving index-production.html by default."""
        if self.path == '/':
            self.path = '/index-production.html'
        return super().do_GET()

def serve_production_dashboard(port=3000, auto_open=False):
    """Serve the production dashboard on the specified port."""
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    # Check if production HTML exists
    if not (frontend_dir / 'index-production.html').exists():
        print("❌ Error: index-production.html not found!")
        print("Please make sure the production frontend file exists.")
        return
    
    # Create server
    try:
        with socketserver.TCPServer(("", port), CORSHTTPRequestHandler) as httpd:
            print("🚀 MedBillGuard Production Dashboard")
            print(f"📍 URL: http://localhost:{port}")
            print(f"📁 Serving from: {frontend_dir.absolute()}")
            print("🌐 Connected to Railway: https://endearing-prosperity-production.up.railway.app")
            print("⏹️  Press Ctrl+C to stop the server")
            print("-" * 60)
            
            if auto_open:
                webbrowser.open(f"http://localhost:{port}")
            
            httpd.serve_forever()
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {port} is already in use. Try a different port:")
            print(f"   python serve-production.py {port + 1}")
        else:
            print(f"❌ Error starting server: {e}")
    except KeyboardInterrupt:
        print("\n👋 Production dashboard server stopped")

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    auto_open = '--open' in sys.argv
    serve_production_dashboard(port, auto_open) 