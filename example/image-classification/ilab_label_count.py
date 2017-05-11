import codecs
import os
import numpy as np

f_in = open("trainImageLists.txt",'r')
# f_out = open ("ilab_test.lst","w")
cat_list = ['car','f1car','heli','mil','monster','pickup','plane','semi','tank','van']
# class_id = range(10)
img_ind = 0
#labels_int = np.array()
labels_int = []

for line in f_in:
        line = line.strip()
        #Call that default naming function
        l_parts = line.split('-')
        l_parts_no_letter = [p[1:] for p in l_parts]
        l_parts_int= map(int,l_parts_no_letter[1:7])
        cat_ind = cat_list.index(l_parts[0])
        cat_ind = int(cat_ind)

        labels_int.append([cat_ind]+l_parts_int)
        # f_out.write(img_ind+"\t"cat_ind+"\t".join(l_parts_no_letter[1:]+"\t"+line+"\n"))
        # f_out.write(str(img_ind)+"\t"+str(cat_ind)+"\t"+str(l_parts_int[0])+"\t"+str(l_parts_int[1])+"\t"+str(l_parts_int[2])+"\t"+str(l_parts_int[3])+"\t"+str(l_parts_int[4])+"\t"+str(l_parts_int[5])+"\t"+line+"\n")
        img_ind = img_ind + 1

labarray_train=np.array(labels_int)
f_in.close()

f_in = open("testImageLists.txt",'r')
# f_out = open ("ilab_test.lst","w")
cat_list = ['car','f1car','heli','mil','monster','pickup','plane','semi','tank','van']
# class_id = range(10)
img_ind = 0
labels_int = []
for line in f_in:
        line = line.strip()
        #Call that default naming function
        l_parts = line.split('-')
        l_parts_no_letter = [p[1:] for p in l_parts]
        l_parts_int= map(int,l_parts_no_letter[1:7])
        cat_ind = cat_list.index(l_parts[0])
        cat_ind = int(cat_ind)

        labels_int.append([cat_ind]+l_parts_int)
        # f_out.write(img_ind+"\t"cat_ind+"\t".join(l_parts_no_letter[1:]+"\t"+line+"\n"))
        # f_out.write(str(img_ind)+"\t"+str(cat_ind)+"\t"+str(l_parts_int[0])+"\t"+str(l_parts_int[1])+"\t"+str(l_parts_int[2])+"\t"+str(l_parts_int[3])+"\t"+str(l_parts_int[4])+"\t"+str(l_parts_int[5])+"\t"+line+"\n")
        img_ind = img_ind + 1

labarray_test=np.array(labels_int)
f_in.close()

labarray=np.concatenate((labarray_train,labarray_test),axis=0)
num_classes=[]
num_classes_train=[]
num_classes_test=[]
for i in range(7):
        num_classes.append(len(np.unique(labarray[:,i])))
        num_classes_train.append(len(np.unique(labarray_train[:,i])))
        num_classes_test.append(len(np.unique(labarray_test[:,i])))

print num_classes
print num_classes_test
print num_classes_train        




# f_out.close()

# import codecs
# import os +"\t".join(l_parts_no_letter[1:7])

# f_in = open("trainImageLists.txt",'r')
# f_out = open ("test_out.txt","w")
# for line in f_in:
#         line = line.strip()
#         #Call that default naming function
#         l_parts = line.split('-')
#         l_parts_no_letter = [p[1:] for p in l_parts]
#         f_out.write(l_parts[0]+"\t".join(l_parts_no_letter[1:])+"\t"+line+"\n")
# f_in.close()
# f_out.close()
