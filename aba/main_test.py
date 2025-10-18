"""
Main entry point for generating and analyzing an ABA+ framework.

This script:
  1. Builds an ABA framework from a text specification.
  2. Prints the original (classical) ABA framework.
  3. Prepares the framework for ABA+ (atomic transformation + argument/attack generation).
  4. Generates ABA+ components (assumption combinations, normal/reverse attacks).
  5. Prints the resulting ABA+ framework components.
  6. Plots the ABA+ attack graph between sets of assumptions.
"""

from copy import deepcopy
from aba.aba_builder import build_aba_framework, prepare_aba_plus_framework
from aba.aba_utils import print_aba_plus_results
from aba.aba_framework import ABAFramework


def testABA(aba_framework: ABAFramework):

    copy_framework = deepcopy(aba_framework)

    transformed_framework: ABAFramework = copy_framework.transform_aba()
    print("\n ------- Transformed ABA framework -------\n ")
    print(transformed_framework)

    # Generate arguments
    transformed_framework.generate_arguments()
    gen_args = transformed_framework.arguments
    print("\n ------- Generated arguments -------\n ")
    print(gen_args)

    # Generate attacks
    transformed_framework.generate_attacks()
    attacks = transformed_framework.attacks
    print("\n ------- Generated attacks -------\n ")
    print(attacks, "\n")


def testABAPlus(aba_framework: ABAFramework):

    # === Step 2: Prepare the framework for ABA+ ===

    aba_framework: ABAFramework = prepare_aba_plus_framework(aba_framework)

    # === Step 3: Generate ABA+ components ===
    print("\n" + "=" * 50)
    print("Generating ABA+ Components")
    print("=" * 50)
    aba_framework.make_aba_plus()

    # === Step 4: Print ABA+ results ===
    print_aba_plus_results(aba_framework)
    return aba_framework


def main():
    """
    Main function to generate and analyze an ABA+ framework.
    """
    # === Step 1: Build the ABA framework from input file ===
    print("\n" + "=" * 50)
    print("Building ABA+ Framework")
    print("=" * 50)

    # Build framework from the given input specification file
    aba_framework = build_aba_framework("aba\examples\exemple.txt")
    print(f"\n ------- Original ABA framework -------\n{aba_framework}")

    base_framework = deepcopy(aba_framework)
    testABA(base_framework)

    aba_for_plus = deepcopy(aba_framework)
    testABAPlus(aba_for_plus)


if __name__ == "__main__":
    main()
