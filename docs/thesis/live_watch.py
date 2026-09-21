import os
import sys
import time
import subprocess
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

WATCH_DIR = "/home/simao/Documents/Repositories/Cloud-Compression/docs/thesis"
EXTENSIONS = {'.tex', '.bib', '.png', '.jpg', '.pdf'}

def get_mtimes():
    mtimes = {}
    for root, _, files in os.walk(WATCH_DIR):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in EXTENSIONS and not f.endswith('main.pdf'):
                path = os.path.join(root, f)
                try:
                    mtimes[path] = os.path.getmtime(path)
                except OSError:
                    pass
    return mtimes

def compile_latex():
    print("\n[⚡ Live-Compile] Cambio detectado. Recompilando LaTeX...")
    cmd = ["pdflatex", "-interaction=nonstopmode", "main.tex"]
    res = subprocess.run(cmd, cwd=WATCH_DIR, capture_output=True, text=True)
    if res.returncode == 0 or os.path.exists(os.path.join(WATCH_DIR, "main.pdf")):
        print(f"[✅ Live-Compile] PDF actualizado con éxito ({time.strftime('%H:%M:%S')}) -> http://localhost:8000/viewer.html")
    else:
        print("[❌ Live-Compile] Error al compilar:")
        print(res.stdout[-1000:] if res.stdout else res.stderr[-1000:])

def run_server():
    class CORSHandler(SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            super().end_headers()

    server = HTTPServer(('127.0.0.1', 8000), lambda *args, **kwargs: CORSHandler(*args, directory=WATCH_DIR, **kwargs))
    print("[🌐 Server] Servidor activo en http://localhost:8000/viewer.html")
    server.serve_forever()

# HTML Viewer con Auto-Reload automático
VIEWER_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Tesis LaTeX - Vista en Vivo</title>
    <style>
        body, html { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #2c3e50; font-family: sans-serif; }
        #header { height: 40px; background: #1a252f; color: #ecf0f1; display: flex; align-items: center; justify-content: space-between; padding: 0 15px; }
        #status { font-size: 14px; color: #2ecc71; font-weight: bold; }
        #frame { width: 100%; height: calc(100% - 40px); border: none; }
    </style>
</head>
<body>
    <div id="header">
        <span>📄 Tesis: Compresión de Nubes de Puntos</span>
        <span id="status">● Sincronizado en vivo</span>
    </div>
    <iframe id="frame" src="/main.pdf#view=FitH"></iframe>

    <script>
        let lastModified = null;
        const frame = document.getElementById('frame');
        const status = document.getElementById('status');

        async function checkUpdate() {
            try {
                const res = await fetch('/main.pdf', { method: 'HEAD', cache: 'no-cache' });
                const cur = res.headers.get('Last-Modified') || res.headers.get('ETag');
                if (lastModified && cur && cur !== lastModified) {
                    status.style.color = '#f39c12';
                    status.innerText = '⚡ Actualizando PDF...';
                    frame.src = '/main.pdf?t=' + Date.now() + '#view=FitH';
                    setTimeout(() => {
                        status.style.color = '#2ecc71';
                        status.innerText = '● Sincronizado (' + new Date().toLocaleTimeString() + ')';
                    }, 800);
                }
                lastModified = cur;
            } catch (e) {}
        }
        setInterval(checkUpdate, 1500);
    </script>
</body>
</html>
"""

if __name__ == '__main__':
    with open(os.path.join(WATCH_DIR, "viewer.html"), "w") as f:
        f.write(VIEWER_HTML)

    t = threading.Thread(target=run_server, daemon=True)
    t.start()

    compile_latex()
    last_mtimes = get_mtimes()

    print("[👀 Live-Watcher] Observando cambios en docs/thesis/*.tex...")
    while True:
        time.sleep(1)
        cur_mtimes = get_mtimes()
        if cur_mtimes != last_mtimes:
            compile_latex()
            last_mtimes = cur_mtimes
