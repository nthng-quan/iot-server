import http.server
import socketserver

PORT = 5551

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()   