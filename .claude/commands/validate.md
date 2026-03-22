Run field contract validation: `make validate-fields`. Then check the output against the contract in `src/validation/field_contract.cpp`. Report:
- Total contract fields
- Exported count
- NotImplemented count
- RequiredNow coverage
- Any strict-mode violations
If violations exist, trace them to the source module and suggest fixes.