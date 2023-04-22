from multiprocessing import Process
a=[]
c=[]
def add(b,d):
    a.append(b)
    print(a)
    d=a
if __name__ == '__main__':
    process_list=[]
    p1=Process(target=add,args=(1,c))
    p1.start()
    process_list.append(p1)
    p2=Process(target=add,args=(2,c))
    p2.start()
    process_list.append(p2)    
    p3=Process(target=add,args=(3,c))
    p3.start()
    process_list.append(p3)
    
    p1.join()
    p2.join()
    
    p3.join()
    print(c)