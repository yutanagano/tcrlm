# CDR3 encoding

A repository for the CDR3 encoding project.

## Notes on the amino acid tuplet vocabulary system

Because of this codebase's modular tokenisation system, the models' vocabulary
can change depending on how the tokenistaion is set. However, to keep things
consistent, there are always two tokens that get mapped to the same index:

- The padding token, which is always `0`
- The mask token, which is always `1`