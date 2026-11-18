"""Build a temporary Linux headless Engine Simulator and render its default engine."""
from pathlib import Path
import argparse
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path, help='Engine Simulator checkout with initialized submodules')
    parser.add_argument('--output', type=Path, default=PROJECT / 'demos/engine_sim_subaru_free_rev.wav')
    args = parser.parse_args()
    source = args.checkout.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f'Output already exists: {output}; choose another path')
    with tempfile.TemporaryDirectory(prefix='engine-sim-headless-') as work:
        root = Path(work) / 'source'
        root.mkdir()
        for folder in ('src', 'include', 'scripting', 'assets', 'es'):
            shutil.copytree(source / folder, root / folder)
        for sub in ('piranha', 'simple-2d-constraint-solver', 'csv-io'):
            shutil.copytree(source / 'dependencies/submodules' / sub, root / 'dependencies/submodules' / sub,
                            ignore=shutil.ignore_patterns('.git'))
        for filename in ('CMakeLists.txt', 'common.h', 'headless.cpp'):
            shutil.copy2(HERE / filename, root / filename)
        gtest = root / 'gtest'
        gtest.mkdir()
        (gtest / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.10)
find_package(GTest CONFIG REQUIRED)
foreach(t IN ITEMS gtest gtest_main gmock gmock_main)
  add_library(${t} INTERFACE)
  target_link_libraries(${t} INTERFACE GTest::${t})
endforeach()
''')
        subprocess.run(['git', 'apply', '--unsafe-paths', str(HERE / 'linux-port.patch')], cwd=root, check=True)
        build = Path(work) / 'build'
        subprocess.run(['cmake', '-S', str(root), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release', '-Wno-dev'], check=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'headless', '-j', '4'], check=True)
        subprocess.run([str(build / 'headless'), str(output)], cwd=root, check=True)
        print(f'Rendered: {output}')


if __name__ == '__main__':
    run()
