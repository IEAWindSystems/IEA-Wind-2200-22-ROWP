# usage: python examples/validate.py
import windIO
schema_type='plant/wind_energy_system'
windIO.validate(input='inputs/wind_energy_system.yaml', schema_type=schema_type)
print('validation successful')
