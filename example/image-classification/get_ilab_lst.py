import codecs
import os

f_in = open("testImageLists.txt",'r')
f_out = open ("ilab_test.lst","w")
cat_list = ['car','f1car','heli','mil','monster','pickup','plane','semi','tank','van']
# class_id = range(10)
img_ind = 0
for line in f_in:
        line = line.strip()
        #Call that default naming function
        l_parts = line.split('-')
        l_parts_no_letter = [p[1:] for p in l_parts]
        l_parts_int= map(int,l_parts_no_letter[1:7])

        cat_ind = cat_list.index(l_parts[0])
        cat_ind = int(cat_ind)
        # f_out.write(img_ind+"\t"cat_ind+"\t".join(l_parts_no_letter[1:]+"\t"+line+"\n"))
 		# f_out.write(str(img_ind)+"\t"+str(cat_ind)+"\t"+str(l_parts_int[0])+"\t"+str(l_parts_int[1])+"\t"+str(l_parts_int[2])+"\t"+str(l_parts_int[3])+"\t"+str(l_parts_int[4])+"\t"+str(l_parts_int[5])+"\t"+line+"\n")
        #only good subset 0,2,3,4 (complete) -- 1,2,3 (lparts_int)
        f_out.write(str(img_ind)+"\t"+str(cat_ind)+"\t"+str(l_parts_int[1])+"\t"+str(l_parts_int[2])+"\t"+str(l_parts_int[3])+"\t"+line+"\n")

        img_ind = img_ind + 1

f_in.close()
f_out.close()

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
