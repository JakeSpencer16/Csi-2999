
num_years = 30
# Create a list of years from 1994 to 2023


years = list(range(1994, 2024)) # 30 years

dsv_array1 = [100 + i * 2 for i in range(30)]  #  steadily increasing
dsv_array2 = [120 - i for i in range(30)]      # steadily decreasing
dsv_array3 = [100 + ((-1) ** i) * (i % 5) for i in range(30)]  # fluctuating

print('years:', years)
print('DSV Array 1:', dsv_array1)
print('DSV Array 2:', dsv_array2)
print('DSV Array 3:', dsv_array3)
