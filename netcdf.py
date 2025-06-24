from netCDF4 import Dataset

if __name__ == "__main__":
    # Open the netCDF file
    nc_file = Dataset("data/reanalysis/air.2025.nc")

    # Print the variables in the file
    print(nc_file.variables.keys())

    # Print the dimensions in the file
    print(nc_file.dimensions.keys())
