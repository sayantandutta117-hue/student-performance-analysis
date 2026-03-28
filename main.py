#student analysis report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import sys
def main():
    n= int(input("enter the number of students::"))
    name =  []
    roll=[]
    marks=[]
    phone= []
    for i in range(n):
       student_name= input("enter student name:")
       student_roll= int(input("enter student roll::"))
       student_total_marks= int(input("enter total marks:"))
       student_phone_number= int(input("enter student phone number::"))
       if len(str(student_phone_number))!=10:
           print("reenter the phone number")
           student_phone_number = int(input("enter student phone number::"))
       name.append(student_name)
       roll.append(student_roll)
       marks.append(student_total_marks)
       phone.append(student_phone_number)
    student= pd.DataFrame({"name":name,"roll":roll,"marks":marks,"phone":phone})
    print(student)
    return student
def print_max_marks_student(student):
        if student.empty:
            print("no student available")
            return
        row= student.loc[student["marks"].idxmax()]
        print(row["roll"])
        print(row["name"])

def add(student):
    u= input("if you want to add new students::")
    if u.lower()=="y" or u.lower()=="yes":
        n= int(input("enter how many students you want to add::"))
        name= []
        roll= []
        marks= []
        phone= []
        for i in range(n):
             new_student_name= input("enter student name:")
             new_student_roll= int(input("enter student roll::"))
             new_student_total_marks= int(input("enter total marks:"))
             new_student_phone_number= int(input("enter student phone number::"))
             if len(str(new_student_phone_number))!=10:
                 print("reenter the phone number")
                 new_student_phone_number = int(input("enter student phone number::"))
             name.append(new_student_name)
             roll.append(new_student_roll)
             marks.append(new_student_total_marks)
             phone.append(new_student_phone_number)
        new_student= pd.DataFrame({"name":name,"roll":roll,"marks":marks,"phone":phone})
        student=pd.concat([student,new_student],ignore_index=True)
        return student
    elif u.lower()=="n" or u.lower()=="no":
        print("thank you , no new students")
    return student
def print_min_marks_student(student):
    if student.empty:
        print("No student available")
        return
    row= student.loc[student["marks"].idxmin()]
    print(row["roll"])
    print(row["name"])

def update(student):
    n= input("want to update::y/n")
    if n.lower()=="y" or n.lower()=="yes":
        r= int(input("enter roll::"))
        if r not in student["roll"].values:
            print("no such student present")
            return student
        else:
           print("what do you want to update::")
           print("1.update roll")
           print("2.update name")
           print("3.update marks")
           print("4.update phone")
           choice= int(input("enter choice::"))
           if choice==1:
               new_r= int(input("enter new roll::"))
               student.loc[student["roll"]==r,"roll"]=new_r
           elif choice==2:
               new_n= input("enter new name::")
               student.loc[student["roll"]==r,"name"]=new_n
           elif choice==3:
               new_m= int(input("enter new marks:::"))
               student.loc[student["roll"]==r,"marks"]=new_m
           elif choice==4:
               new_ph= int(input("enter new phone number::"))
               student.loc[student["roll"]==r,"phone"]=new_ph
    elif n.lower()=="n" or n.lower()=="no":
        print("everything is correct")
    print(student)
    return student
def average(student):
    if student.empty:
        print("no student available")
        return
    p= student["marks"].mean()
    print("Average marks::",p)
def visualize(student):
    if student.empty:
        print("no student available")
        return
    x_axis= student["name"]
    y_axis= student["marks"]
    plt.bar(x_axis,y_axis,color="green")
    plt.xlabel("student name",color="RED",fontsize=15)
    plt.ylabel("student marks",color="BLUE",fontsize=15)
    plt.title("Students performance",color="YELLOW",fontsize=15)
    plt.show()
def show_all_students(student):
    if student.empty:
        print("no student available")
        return
    print(student)
def ranking(student):
    if student.empty:
        print("no student available")
        return
    s= student.sort_values(by="marks",ascending=False)
    print(s)
student = pd.DataFrame(columns=["name","roll","marks","phone"])
while True:
    print("1. PREPARING STUDENTS DETAILS")
    print("2. ADDING NEW STUDENTS")
    print("3. FINDING STUDENT WITH MAXIMUM MARKS")
    print("4. FINDING STUDENT WITH MINIMUM MARKS")
    print("5. UPDATING STUDENT")
    print("6. FINDING AVERAGE MARKS")
    print("7. VISUALISING STUDENT PERFMANCE")
    print("8. SHOW ALL THE STUDENTS")
    print("9. RANKING THE STUDENTS")
    print("10. EXIT")
    k= int(input("enter your choice::"))
    if k==1:
        student = main()
    elif k==2:
        student = add(student)
    elif k==3:
        print_max_marks_student(student)
    elif k==4:
        print_min_marks_student(student)
    elif k==5:
        student = update(student)
    elif k==6:
        average(student)
    elif k==7:
        visualize(student)
    elif k==8:
        show_all_students(student)
    elif k==9:
        ranking(student)
    elif k==10:
        sys.exit()


