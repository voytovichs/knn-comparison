import signal
import sys
import waitress
from vectorsindex import app


def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)
waitress.serve(host='0.0.0.0', port=80, app=app)
