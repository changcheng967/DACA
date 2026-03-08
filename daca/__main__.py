"""DACA command line interface.

Usage:
    python -m daca [command]

Commands:
    info      Show DACA information
    patch     Apply DACA patches
    probe     Run hardware probe
    doctor    Run environment diagnostics
    bench     Run benchmarks
"""

import sys
import argparse


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="daca",
        description="DACA - DaVinci Accelerated Compute Architecture"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show DACA information")
    info_parser.set_defaults(func=cmd_info)

    # Patch command
    patch_parser = subparsers.add_parser("patch", help="Apply DACA patches")
    patch_parser.set_defaults(func=cmd_patch)

    # Probe command
    probe_parser = subparsers.add_parser("probe", help="Run hardware probe")
    probe_parser.add_argument("--output", "-o", type=str, default="probe_data.json",
                              help="Output JSON file")
    probe_parser.set_defaults(func=cmd_probe)

    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run environment diagnostics")
    doctor_parser.set_defaults(func=cmd_doctor)

    # Benchmark command
    bench_parser = subparsers.add_parser("bench", help="Run benchmarks")
    bench_parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    bench_parser.set_defaults(func=cmd_bench)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


def cmd_info(args):
    """Run info command."""
    import daca
    daca.info()
    return 0


def cmd_patch(args):
    """Run patch command."""
    import daca

    print("Applying DACA patches...")
    daca.patch()
    print("Patches applied. Press Ctrl+C to exit.")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nRemoving patches...")
        daca.unpatch()
        print("Done.")

    return 0


def cmd_probe(args):
    """Run probe command."""
    from daca.tools.probe import main as probe_main

    sys.argv = ["daca probe", "--output", args.output]
    return probe()


def cmd_doctor(args):
    """Run doctor command."""
    from daca.tools.doctor import main as doctor_main
    return doctor_main()


def cmd_bench(args):
    """Run benchmark command."""
    from daca.autotune.benchmark import run_all_benchmarks

    results = run_all_benchmarks(verbose=True)

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
