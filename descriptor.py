import pandas


descriptor_values = pandas.read_csv('/home/berkay/Desktop/EndDescriptors/esf/DescriptorValuesESF.csv')
descriptor_values = descriptor_values.drop(descriptor_values.columns[640], axis = 1)
print(descriptor_values)
