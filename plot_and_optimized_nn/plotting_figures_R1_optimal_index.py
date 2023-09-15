# from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
## Written by Haiying Yang
## Date: June,2021

# nn_parameter_data=np.loadtxt('progress2.txt', comments='#', delimiter=' '）
population_size=15

nn_parameter_data=np.loadtxt('progress_forwardnn.txt', comments='#', delimiter=' ')       

f_best=np.loadtxt('f_best.txt', comments='#', delimiter=' ')       

B=np.where(nn_parameter_data[:,7]==f_best[0])[0]

ncolumn=np.size(nn_parameter_data,1)

output=nn_parameter_data[:,(ncolumn-1)]

NCHROM=np.size(nn_parameter_data,0)

GENERATION=int(NCHROM/population_size)
best_ind_total=[]

optimized_index=np.where(output==min(f_best))[0]+1         #次序从零开始未加1

print('forward nn 最优值index:%s' %optimized_index)   


#finding scaled output
for generation in range(0,GENERATION):
    best_ind= np.where(output==f_best[generation])[0]
    best_ind_total=np.append(best_ind_total,best_ind)
    
    best_ind_total=best_ind_total.astype(int)


optimized_cost=f_best
generation_plot=np.arange(1,GENERATION+1)

#######scaled
##optimized_output vs. Generation
plt.plot(generation_plot,optimized_cost ,'b-s')
plt.xlabel('Generation')
plt.ylabel('optimized_cost') 
plt.title('Optimized cost vs. Generation') 
plt.savefig("Optimized cost vs. Generation.png") 
plt.show()
plt.close() 




E_test=nn_parameter_data[best_ind_total[:],8]

##E_test_scaled vs. Generation

plt.plot(generation_plot,E_test ,'r-o')
plt.xlabel('Generation')
plt.ylabel('E_surr') 
plt.title('E_surr vs. Generation')
plt.savefig("E_surr vs. Generation.png")          
plt.show()
plt.close() 

   

overfitting=nn_parameter_data[best_ind_total[:],7]

##E_test vs. Generation

plt.plot(generation_plot,overfitting,'k-^')
plt.xlabel('Generation')
plt.ylabel('overfitting') 
plt.title('overfitting vs. Generation')  
plt.savefig("overfitting vs. Generation.png")
plt.show()
plt.close() 

FLOPS=nn_parameter_data[best_ind_total[:],9]

##E_test vs. Generation

plt.plot(generation_plot,FLOPS,'m->')
plt.xlabel('Generation')
plt.ylabel('FLOPS') 
plt.title('FLOPS vs. Generation')  
plt.savefig("FLOPS vs. Generation.png")
plt.show()
plt.close() 


##E_test,overfitting quantity, FLOPS

#plt.plot(generation_plot,FLOPS,'m->',label='FLOPS',generation_plot,overfitting,'k-^',generation_plot,E_test ,'r-o')
l1,=plt.plot(generation_plot,FLOPS,'m->',label='FLOPS')
l2,=plt.plot(generation_plot,overfitting,'k-^',label='Overfitting')
l3,=plt.plot(generation_plot,E_test,'r-o',label='E_surr')

plt.legend()

plt.xlabel('Generation')
plt.ylabel('FLOPS & Overfitting quantity & E_surr') 
plt.title('FLOPS & Overfitting quantity & E_surr vs. Generation')  

plt.savefig("FLOPS & Overfitting quantity & E_surr vs. Generation.png")
plt.show()
plt.close() 







#######unscaled




E_test_unscaled=nn_parameter_data[best_ind_total[:],4]

##E_test_scaled vs. Generation

plt.plot(generation_plot,E_test_unscaled,'r-o')
plt.xlabel('Generation')
plt.ylabel('Unscaled testing error') 
plt.title('Unscaled testing error vs. Generation')
plt.savefig("Unscaled testing error vs. Generation.png")          
plt.show()
plt.close() 

   

overfitting_unscaled=nn_parameter_data[best_ind_total[:],3]

##E_test vs. Generation

plt.plot(generation_plot,overfitting_unscaled,'k-^')
plt.xlabel('Generation')
plt.ylabel('Unscaled overfitting') 
plt.title('Unscaled overfitting vs. Generation')  
plt.savefig("Unscaled overfitting vs. Generation.png")
plt.show()
plt.close() 

FLOPS_unscaled=2*nn_parameter_data[best_ind_total[:],5]

##E_test vs. Generation

plt.plot(generation_plot,FLOPS_unscaled,'m->')
plt.xlabel('Generation')
plt.ylabel('Unscaled FLOPS') 
plt.title('Unscaled FLOPS vs. Generation')  
plt.savefig("Unscaled FLOPS vs. Generation.png")
plt.show()
plt.close() 



##optimized_output_unscaled vs. Generation
optimized_cost_unscaled=E_test_unscaled+overfitting_unscaled+FLOPS_unscaled

#optimized_output_unscaled=nn_parameter_data[best_ind_total[:],6]
plt.plot(generation_plot,optimized_cost_unscaled,'b-s')
plt.xlabel('Generation')
plt.ylabel('Unscaled optimized cost') 
plt.title('Unscaled optimized cost vs. Generation') 
plt.savefig("Unscaled optimized cost vs. Generation.png") 
plt.show()
plt.close() 

