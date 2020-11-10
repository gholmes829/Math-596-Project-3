#!/usr/bin/env python3
import sys
from driver import Driver

def main(argv: list, argc: int) -> None:
	"""
	Put argv, argc here in case we wanna pass anything in via cmd line
	"""
	driver = Driver()
	driver.run()

if __name__ == "__main__":
	main(sys.argv, len(sys.argv))
