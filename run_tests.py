#!/usr/bin/env python3
"""
Test runner for the transcribe package.
"""
import sys
import unittest
import argparse
import coverage

def run_tests(test_pattern=None, coverage_report=False):
    """
    Run tests with optional coverage report.
    
    Args:
        test_pattern (str, optional): Pattern to match test names. Defaults to None.
        coverage_report (bool, optional): Whether to generate coverage report. Defaults to False.
    
    Returns:
        int: 0 if all tests pass, non-zero otherwise
    """
    if coverage_report:
        cov = coverage.Coverage(
            source=['transcribe_pkg'],
            omit=['transcribe_pkg/core/examples.py']
        )
        cov.start()
    
    # Discover and load tests
    loader = unittest.TestLoader()
    if test_pattern:
        tests = loader.loadTestsFromName(test_pattern)
    else:
        tests = loader.discover('tests')
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(tests)
    
    if coverage_report:
        cov.stop()
        cov.save()
        cov.report()
        cov.html_report(directory='coverage_html')
        print(f"HTML coverage report generated in 'coverage_html' directory")
    
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run transcribe package tests')
    parser.add_argument('-p', '--pattern', 
                        help='Pattern to match test names (e.g., "test_config")')
    parser.add_argument('-c', '--coverage', action='store_true',
                        help='Generate coverage report')
    
    args = parser.parse_args()
    
    sys.exit(run_tests(args.pattern, args.coverage))

#fin