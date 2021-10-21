# import pandas package
import pandas as pd
 
# List of Tuples
students = [('Ankit', 22, 'A'),
           ('Swapnil', 22, 'B'),
           ('Priya', 22, 'B'),
           ('Shivangi', 22, 'B'),
            ]
# Create a DataFrame object
stu_df = pd.DataFrame(students, columns =['Name', 'Age', 'Section'],
                      index =['1', '2', '3', '4'])
 
stu_df

 # Iterate over column names
for column in stu_df:
    print(column)
    # Select column contents by column
    # name using [] operator
    columnSeriesObj = stu_df[column]
# =============================================================================
#     print('Column Name : ', column)
#     print('Column Contents : ', columnSeriesObj.values)
# =============================================================================
