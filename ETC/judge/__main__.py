"""Run with python -m ETC.judge --help from the course repository."""
import argparse
from pathlib import Path
import time

from .store import Store


def main():
    parser = argparse.ArgumentParser(description="Challenge 1-4 local scoring pilot (not an official event deployment)")
    parser.add_argument("--state", type=Path, required=True, help="Private local state directory, outside the teaching repository")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("--steps", type=int, default=200)
    init.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    add = commands.add_parser("add-participant")
    add.add_argument("nickname", nargs="?", help="Optional display name; omit so the participant chooses one in Jupyter")
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8090)
    worker = commands.add_parser("worker")
    worker.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    worker.add_argument("--gpu", type=int)
    worker.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "init":
        Store.initialize(args.state, args.steps, args.device)
        print("Pilot initialized. Add participants, then start serve and a worker in separate terminals.")
        return
    store = Store(args.state)
    store.require_current_version()
    if args.command == "add-participant":
        print("Participant access code (shown once; share privately):")
        print(store.add_participant(args.nickname))
    elif args.command == "serve":
        from .web import create_server
        server = create_server(store, args.port)
        print(f"Local pilot: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    elif args.command == "worker":
        from .worker import work_once
        if args.device == "cuda" and (args.gpu is None or args.gpu < 0):
            parser.error("A CUDA worker requires one nonnegative --gpu index")
        while True:
            worked = work_once(store, args.device, args.gpu)
            if args.once:
                break
            if not worked:
                time.sleep(2)


if __name__ == "__main__":
    main()
