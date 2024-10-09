import sqlite3

## Connectt to SQlite
connection=sqlite3.connect("student.db")

# Create a cursor object to insert record,create table

cursor=connection.cursor()

## create the table
table_info_new="""CREATE TABLE STUDENT (
    ID INTEGER PRIMARY KEY,   
    NAME VARCHAR(25),                       
    CLASS VARCHAR(25),                      
    SECTION VARCHAR(25),                    
    MARKS INT,                              
    AGE INT,                                
    GENDER VARCHAR(10),                     
    ADDRESS TEXT,                           
    CONTACT_NUMBER VARCHAR(15),             
    EMAIL VARCHAR(50),                      
    DATE_OF_BIRTH DATE,                     
    ADMISSION_DATE DATE,                    
    GUARDIAN_NAME VARCHAR(50),              
    GUARDIAN_CONTACT_NUMBER VARCHAR(15),    
    TOTAL_FEES REAL,                        
    FEES_PAID REAL,                         
    PERCENTAGE REAL,                        
    REMARKS TEXT                            
);

"""



cursor.execute(table_info_new)

## Insert Some more records

cursor.execute('''Insert Into STUDENT values(1,'John Doe', '10', 'A', 85, 15, 'Male', '123 Main St', '1234567890', 'john.doe@example.com', '2009-05-14', '2024-01-10', 'Jane Doe', '0987654321', 1500.00, 1500.00, 85.0, 'Good performance')''')
cursor.execute('''Insert Into STUDENT values(2,'Alice Smith', '9', 'B', 90, 14, 'Female', '456 Oak St', '2345678901', 'alice.smith@example.com', '2010-08-21', '2024-01-12', 'Bob Smith', '1231231234', 1600.00, 1600.00, 90.0, 'Excellent performance')''')
cursor.execute('''Insert Into STUDENT values(3,'Bob Johnson', '11', 'A', 78, 16, 'Male', '789 Pine St', '3456789012', 'bob.johnson@example.com', '2008-03-09', '2024-01-15', 'Sara Johnson', '3213213210', 1400.00, 1000.00, 78.0, 'Needs improvement')''')
cursor.execute('''Insert Into STUDENT values(4,'Catherine Brown', '12', 'C', 88, 17, 'Female', '987 Cedar St', '4567890123', 'catherine.brown@example.com', '2007-07-13', '2024-01-20', 'Tom Brown', '4324324321', 1800.00, 1800.00, 88.0, 'Consistent performance')''')
cursor.execute('''Insert Into STUDENT values(5,'David Wilson', '10', 'A', 92, 15, 'Male', '654 Birch St', '5678901234', 'david.wilson@example.com', '2009-11-22', '2024-01-22', 'Mary Wilson', '5435435432', 1500.00, 1500.00, 92.0, 'Outstanding')''')
cursor.execute('''Insert Into STUDENT values(6,'Emily Davis', '9', 'C', 81, 14, 'Female', '321 Elm St', '6789012345', 'emily.davis@example.com', '2010-06-30', '2024-01-25', 'Mike Davis', '6546546543', 1600.00, 1400.00, 81.0, 'Good work ethic')''')
cursor.execute('''Insert Into STUDENT values(7,'Frank Miller', '11', 'B', 76, 16, 'Male', '789 Maple St', '7890123456', 'frank.miller@example.com', '2008-02-17', '2024-01-27', 'Rachel Miller', '7657657654', 1700.00, 1700.00, 76.0, 'Needs more focus')''')
cursor.execute('''Insert Into STUDENT values(8,'Grace Lee', '12', 'A', 94, 17, 'Female', '111 Aspen St', '8901234567', 'grace.lee@example.com', '2007-04-05', '2024-02-02', 'Harry Lee', '8768768765', 1800.00, 1800.00, 94.0, 'Top performer')''')
cursor.execute('''Insert Into STUDENT values(9,'Henry Martinez', '10', 'B', 80, 15, 'Male', '222 Redwood St', '9012345678', 'henry.martinez@example.com', '2009-09-10', '2024-02-05', 'Lisa Martinez', '9879879876', 1500.00, 1300.00, 80.0, 'Shows potential')''')
cursor.execute('''Insert Into STUDENT values(10,'Isabella Thompson', '9', 'A', 87, 14, 'Female', '333 Walnut St', '0123456789', 'isabella.thompson@example.com', '2010-10-25', '2024-02-08', 'James Thompson', '1098765432', 1600.00, 1600.00, 87.0, 'Very promising')''')



## Disspaly ALl the records

print("The isnerted records are")
data=cursor.execute('''Select * from STUDENT''')
for row in data:
    print(row)

## Commit your changes int he databse
connection.commit()
connection.close()