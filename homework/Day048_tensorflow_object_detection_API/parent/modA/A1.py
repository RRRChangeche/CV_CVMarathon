def A1_function():
	print("Hello A1 function!")


#if __name__ == "__main__":
import os 
print(os.getcwd())
# os.chdir("parent")
# print(os.getcwd())
import sys
sys.path.append("parent/modA")
sys.path.append("parent/modB")
sys.path.append("parent")

from A2 import A2_function as A2F
A2F()

from parent.modB.B1 import B1_function as B1F
B1F()