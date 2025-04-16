import pymysql
import chardet
import numpy as np
import re
import os

source_dir = "K-means/data/source"
# 连接到 MySQL 数据库
conn = pymysql.connect(
    host="10.80.42.230",
    user="hwcheck",
    password="hw_CheCk-For24251*oOP",
    database="homework",
    charset="gbk"
)

# 创建游标
cursor = conn.cursor()



class Student:
    def __init__(self, stu_no, stu_name, stu_cno):
        self.no = stu_no
        self.name = stu_name
        self.cno = stu_cno

    def is_match(self, other):
        if self.no == other.no or self.name == other.name:
            return True
        else:
            return False
        
    def output(self):
        print(f"{self.no}, {self.name}, {self.cno}")
        

class StudentGraph:
    def __init__(self):
        self.students = []
        self.no_to_index = {}
        self.stu_no = []
        self.edges = []

    def add_node(self, stu_no, related_students):
        self.stu_no.append(stu_no)
        self.edges.append([s.no for s in related_students])

    def output(self):
        print("Students:")
        for stu in self.students:
            stu.output()
        print("Nodes:")
        for i in range(len(self.stu_no)):
            print(f"{i}. {self.stu_no[i]}, edges: {self.edges[i]}")

    def self_check(self):
        print(f"students num: {len(self.students)}")
        print(f"nodes num: {len(self.stu_no)}, {len(self.edges)}")

        for i in range(len(self.stu_no)):
            for no in self.edges[i]:
                if no not in self.no_to_index:
                    print(f"Error: {no} not found in no_to_index.")

    def Symmetry(self): # 通过添加边使图变为对称
        for i in range(len(self.stu_no)):
            cur_stu_no = self.stu_no[i]
            related_stu_no_indexs = [self.no_to_index[s_no] for s_no in self.edges[i]]
            for j in range(len(related_stu_no_indexs)):
                if cur_stu_no not in self.edges[related_stu_no_indexs[j]]:
                    self.edges[related_stu_no_indexs[j]].append(cur_stu_no)

def delete_hang_node(gra):
    new_gra = StudentGraph()
    count = 0
    for i in range(len(gra.stu_no)):
        if len(gra.edges[i]) == 0:
            continue
        new_gra.stu_no.append(gra.stu_no[i])
        new_gra.edges.append(gra.edges[i])
        new_gra.no_to_index[gra.stu_no[i]] = count
        new_gra.students.append(gra.students[i])
        count += 1
    
    new_gra.self_check()
    return new_gra


def check_student(student, cursor):
    # 执行 SQL 查询
    cursor.execute(f"SELECT stu_no, stu_name, stu_cno FROM view_hwcheck_stulist WHERE stu_no = '{student.no}'")
    # 获取查询结果
    results = cursor.fetchall()
    if results:
        student.no, student.name, student.cno = results[0]
        return True
    else: # 尝试匹配姓名
        cursor.execute(f"SELECT stu_no, stu_name, stu_cno FROM view_hwcheck_stulist WHERE stu_name = '{student.name}'")
        results = cursor.fetchall()
        if results:
            student.no, student.name, student.cno = results[0]
            return True
        else:
            return False


def read_second_non_empty_line(file_path, enc):
    with open(file_path, 'r', encoding=enc) as file:
        non_empty_line_count = 0  # 记录非空行的数量
        for line in file:
            # 去除行首尾的空白字符（包括空格、换行符等）
            stripped_line = line.strip()
            # 如果行不为空
            if stripped_line:
                non_empty_line_count += 1
                # 如果是第二行非空行，返回该行内容
                if non_empty_line_count == 2:
                    return stripped_line
    # 如果文件中没有第二行非空行，返回 None
    return None

def get_text(file_dir):
    try:
        text = read_second_non_empty_line(file_dir, 'gbk')
    except UnicodeDecodeError:
        try:
            text = read_second_non_empty_line(file_dir, 'utf-8')
        except UnicodeDecodeError:
            print(f"{file_dir}: unknown encoding.")
            text = None
    return text



def get_students(file_dir, cno):
    if not os.path.exists(file_dir):
        print(f"{file_dir} not found.")
        return []
    
    text = get_text(file_dir)
    if text is None:
        return []
    
    print(f"Get content: {text[:100]}... from {file_dir}.")

    if text.startswith("//"):
        content = text[2:].strip()  # 去掉 "//" 并去除前后空格
    # 检查是否被 "/*" 和 "*/" 包围
    elif re.match(r'/\*.*\*/', text):
        content = re.sub(r'/\*|\*/', '', text).strip()  # 去掉 "/*" 和 "*/" 并去除前后空格
    else:
        print(f"Error: {file_dir} has no desire content.")
        return []
    
    parts = content.split()
    students = []
    for i in range(0, len(parts), 2):
        if i+1 < len(parts):
            related_stu = Student(parts[i], parts[i+1], cno)
            if not check_student(related_stu, cursor):
                print(f"Error: {related_stu.no} not found in database.")
                continue
            students.append(related_stu)
    
    print(f"Get {len(students)} students from {file_dir}.")
    return students

def get_graph():
    count = 0
    gra = StudentGraph()
    for dir_name in os.listdir(source_dir):
        stu_cno, stu_no = dir_name.split("-")
        if (stu_cno != "10108001" and stu_cno != "10108002"):
            continue
        # 获取当前学生信息
        cur_stu = Student(stu_no, "N", stu_cno)
        if not check_student(cur_stu, cursor):
            print(f"Error: {cur_stu.no} not found in database.")
            continue
        # 获取相关联的学生
        file_dir = os.path.join(source_dir, dir_name, "15-b2.cpp")
        print(f"Get students from {file_dir}...")
        students = get_students(file_dir, stu_cno)

        # 添加当前学生和相关学生到图中
        gra.students.append(cur_stu)
        gra.no_to_index[cur_stu.no] = count
        gra.add_node(cur_stu.no, students)
        count += 1
    gra.Symmetry()
    connected_gra = delete_hang_node(gra)
    return connected_gra


def graph2mat(gra):
    # 构造邻接矩阵
    mat = np.zeros((len(gra.stu_no), len(gra.stu_no)))
    for i in range(len(gra.stu_no)):
        src_index = gra.no_to_index[gra.stu_no[i]]
        for no in gra.edges[i]:
            dst_index = gra.no_to_index[no]
            mat[src_index][dst_index] = 1
    return mat

def get_date():
    gra = get_graph()
    mat = graph2mat(gra)
    with open("K-means/graph.txt", "w") as f:
        for i in range(len(mat)):
            for j in range(len(mat)):
                print(mat[i][j], end=" ", file=f)
            print(file=f)
    return mat

if __name__ == '__main__':
    get_date()
            


            

    

