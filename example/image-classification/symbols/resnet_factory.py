'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx

def Bil_unit(data,name,workspace):
    in1=mx.sym.Convolution(data=data, num_filter=350, kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=False, workspace=workspace, name=name + 'conv_in1')
    in1 = mx.sym.BatchNorm(data=in1, fix_gamma=False, eps=2e-5,  name=name+'bn')
    in1= mx.sym.Activation(data=in1, act_type='relu', name='had_act1')
    in1=mx.sym.reshape(in1,shape=(-1,0,1))
    out=mx.sym.batch_dot(in1,in1,transpose_a=False,transpose_b=True,name=name+'_bilinear')
    out=mx.sym.reshape(out,shape=(0,-1))
    out=mx.sym.reshape(out,shape=(-1,0,14,14))
    return out

def Had_unit(data,in_dim,out_dim,name,workspace,act_type='tanh'):
    in1=mx.sym.Convolution(data=data, num_filter=in_dim, kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=False, workspace=workspace, name=name + 'conv_in1')

    in2=mx.sym.Convolution(data=data, num_filter=in_dim, kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=False, workspace=workspace, name=name + 'conv_in2')
    had=in1*in2
    #had= mx.sym.Activation(data=had, act_type=act_type, name='had_act1')
    #out=mx.sym.Convolution(data=data, num_filter=out_dim, kernel=(1,1), stride=(1,1), pad=(0,0),
     #                              no_bias=False, workspace=workspace, name=name + 'conv_out')
    return had

def MCB_unit(data,out_dim,name,compute_size = 128, ifftflag = True):
    #data1=mx.sym.reshape(data,shape=(-1,1,1,0))
    #data1=mx.sym.reshape(data1,shape=(0,-1))
    data1=data
    S1 = mx.sym.Variable(name+'_s1',init = mx.initializer.Plusminusone(),shape = (1,2048),lr_mult=1)
    H1 = mx.sym.Variable(name+'_h1',init = mx.initializer.Index(out_dim),shape = (1,2048),lr_mult=1)
    S2 = mx.sym.Variable(name+'_s2',init = mx.initializer.Plusminusone(),shape = (1,2048),lr_mult=1)
    H2 = mx.sym.Variable(name+'_h2',init = mx.initializer.Index(out_dim),shape = (1,2048),lr_mult=1)
  
    cs1 = mx.contrib.sym.count_sketch( data = data1,s = S1, h = H1,name= name +'_cs1',out_dim = out_dim, processing_batch_size=32) 
    cs2 = mx.contrib.sym.count_sketch( data = data1,s = S2, h = H2,name=name +'_cs2',out_dim = out_dim,processing_batch_size=32) 
    fft1 = mx.contrib.sym.fft(data = cs1, name=name+'_fft1', compute_size = compute_size) 
    #fft1 = mx.sym.BatchNorm(data=fft1, fix_gamma=False, eps=2e-5, name='bn_fft1')
    #fft1 = mx.sym.Activation(data=fft1, act_type='relu', name='fft1_relu1')
    fft2 = mx.contrib.sym.fft(data = cs2, name=name+'_fft2', compute_size = compute_size) 
    #fft2 = mx.sym.BatchNorm(data=fft2, fix_gamma=False, eps=2e-5, name='bn_fft2')
    ##fft2 = mx.sym.Activation(data=fft2, act_type='relu', name='fft2_relu1')


    c = fft1 * fft2
    if ifftflag:
        ifft = mx.contrib.sym.ifft(data = c, name=name+'_ifft', compute_size = compute_size) 
        #ifft = mx.sym.reshape(ifft,shape=(-1,14,14,out_dim))

        return ifft
    else:
        c = mx.sym.reshape(c,shape=(-1,14,14,out_dim))
        return c
    

def residual_gate(data,name,layer_name,workspace,gate_prefix,image_shape=(3,224,224), gate_act='relu',gate_init=mx.initializer.One(),lr_mult=0.001, wd_mult=0):
    
    pre_shape = data.infer_shape(data=(1,image_shape[0],image_shape[1],image_shape[2]))
    #print pre_shape[1][0][1],pre_shape[1][0][2],pre_shape[1][0][3]
    

    flat=mx.symbol.reshape(data=data,shape=(0,1,-1,1), name=name + layer_name+'flat')
    if gate_prefix is not None:
        gate_name=name+layer_name+'_'+gate_prefix+'_residual_gate'
    else:
        gate_name=name+layer_name+'_residual_gate'
        
    gate = mx.sym.Variable(gate_name, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')
    
    gate=mx.sym.Activation(data=gate, act_type=gate_act, name=name+layer_name+'gate_act')
    
                                 
    flat_elemwise = mx.sym.Convolution(data=flat,weight=gate, num_filter=1, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, workspace=workspace, name=name+layer_name+'flat_elemwise')
    
    target_shape=(0,pre_shape[1][0][1],pre_shape[1][0][2],pre_shape[1][0][3])
                                 
    gated_data=mx.sym.reshape(data=flat_elemwise,shape=target_shape, name=name+layer_name+'elemwise')
    
    return gated_data


def residual_gate_coupled(data1,data2,name,layer_name,workspace,gate_prefix,image_shape=(3,224,224), gate_act='sigmoid',gate_init=mx.initializer.One(),lr_mult=0.001, wd_mult=0):
    
    pre_shape = data1.infer_shape(data=(1,image_shape[0],image_shape[1],image_shape[2]))
    #print pre_shape[1][0][1],pre_shape[1][0][2],pre_shape[1][0][3]
    

    flat_1=mx.symbol.reshape(data=data1,shape=(0,1,-1,1), name=name + layer_name[0]+'flat')
    flat_2=mx.symbol.reshape(data=data2,shape=(0,1,-1,1), name=name + layer_name[1]+'flat')

    if gate_prefix is not None:
        gate_name_1=name+layer_name[0]+'_'+gate_prefix+'_gate'
        gate_name_2=name+layer_name[1]+'_'+gate_prefix+'_gate'
    else:
        gate_name_1=name+layer_name[0]+'_gate'
        gate_name_1=name+layer_name[1]+'_gate'
        
    gate_1 = mx.sym.Variable(gate_name_1, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')
    
    gate_1=mx.sym.Activation(data=gate_1, act_type=gate_act, name=name+layer_name[0]+'gate_act')
    gate_2=1-gate_1
    
                                 
    flat_elemwise_1 = mx.sym.Convolution(data=flat_1,weight=gate_1, num_filter=1, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, workspace=workspace, name=name+layer_name[0]+'flat_elemwise')
    
    flat_elemwise_2 = mx.sym.Convolution(data=flat_2,weight=gate_2, num_filter=1, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, workspace=workspace, name=name+layer_name[1]+'flat_elemwise')
    
    target_shape=(0,pre_shape[1][0][1],pre_shape[1][0][2],pre_shape[1][0][3])
                                 
    gated_data_1=mx.sym.reshape(data=flat_elemwise_1,shape=target_shape, name=name+layer_name[0]+'elemwise')
    gated_data_2=mx.sym.reshape(data=flat_elemwise_2,shape=target_shape, name=name+layer_name[1]+'elemwise')
    
    return gated_data_1,gated_data_2

def residual_double_gate_coupled(data1,data2,name,layer_name,workspace,gate_prefix,image_shape=(3,224,224), gate_act='sigmoid',gate_init=mx.initializer.One(),lr_mult=0.001, wd_mult=0):
    
    pre_shape = data1.infer_shape(data=(1,image_shape[0],image_shape[1],image_shape[2]))
    #print pre_shape[1][0][1],pre_shape[1][0][2],pre_shape[1][0][3]
    

    flat_1=mx.symbol.reshape(data=data1,shape=(0,1,-1,1), name=name + layer_name[0]+'flat')
    flat_2=mx.symbol.reshape(data=data2,shape=(0,1,-1,1), name=name + layer_name[1]+'flat')

    if gate_prefix is not None:
        if len(gate_prefix)==1:
            gate_name_1=name+layer_name[0]+'_'+gate_prefix[0]+'_gate'
            gate_name_2=name+layer_name[1]+'_'+gate_prefix[0]+'_gate'
            gate_name_0=name+layer_name[0]+'_'+gate_prefix[0]+'_residual_gate'
        elif len(gate_prefix)>1:
            gate_name_1=[]
            gate_name_2=[]
            gate_name_0=[]
            for gname in gate_prefix:
                
                gate_name_1.append(name+layer_name[0]+'_'+gname+'_gate')
                gate_name_2.append(name+layer_name[1]+'_'+gname+'_gate')
                gate_name_0.append(name+layer_name[0]+'_'+gname+'_residual_gate')
            
    else:
        gate_name_1=name+layer_name[0]+'_gate'
        gate_name_2=name+layer_name[1]+'_gate'
        gate_name_0=name+layer_name[0]+'_residual_gate'
        
    if gate_prefix is None:
        gate_0 = mx.sym.Variable(gate_name_0, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')
        gate_0=mx.sym.Activation(data=gate_0, act_type=gate_act, name=name+layer_name[0]+'residual_gate_act')
        gate_1 = mx.sym.Variable(gate_name_1, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')

        gate_1=mx.sym.Activation(data=gate_1, act_type=gate_act, name=name+layer_name[0]+'gate_act')
        gate_2=1-gate_1
        gate_1=gate_0*gate_1
        gate_2=gate_0*gate_2
    elif len(gate_prefix)==1:
        gate_0 = mx.sym.Variable(gate_name_0, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')
        gate_0=mx.sym.Activation(data=gate_0, act_type=gate_act, name=name+layer_name[0]+'residual_gate_act')
        gate_1 = mx.sym.Variable(gate_name_1, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')

        gate_1=mx.sym.Activation(data=gate_1, act_type=gate_act, name=name+layer_name[0]+'gate_act')
        gate_2=1-gate_1
        gate_1=gate_0*gate_1
        gate_2=gate_0*gate_2
    elif len(gate_prefix)>1:
        gate_0_list=[]
        gate_1_list=[]
        for g,(g_name_0,g_name_1) in enumerate(zip(gate_name_0,gate_name_1)):
            gate_0_list.append(mx.sym.Variable(g_name_0, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32'))
            gate_1_list.append(mx.sym.Variable(g_name_1, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32'))
        
        gate_0=gate_0_list[0]
        gate_1=gate_1_list[0]
        for ind in range(1,len(gate_prefix)):
            gate_0+=gate_0_list[ind]
            gate_1+=gate_1_list[ind]
    
            
        gate_0=mx.sym.Activation(data=gate_0, act_type=gate_act, name=name+layer_name[0]+'residual_gate_act')
        #gate_1 = mx.sym.Variable(gate_name_1, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')

        gate_1=mx.sym.Activation(data=gate_1, act_type=gate_act, name=name+layer_name[0]+'gate_act')
            
            
        gate_2=1-gate_1    
        gate_1=gate_0*gate_1
        gate_2=gate_0*gate_2
        
       
                                 
    flat_elemwise_1 = mx.sym.Convolution(data=flat_1,weight=gate_1, num_filter=1, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, workspace=workspace, name=name+layer_name[0]+'flat_elemwise')
    
    flat_elemwise_2 = mx.sym.Convolution(data=flat_2,weight=gate_2, num_filter=1, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True, workspace=workspace, name=name+layer_name[1]+'flat_elemwise')
    
    target_shape=(0,pre_shape[1][0][1],pre_shape[1][0][2],pre_shape[1][0][3])
                                 
    gated_data_1=mx.sym.reshape(data=flat_elemwise_1,shape=target_shape, name=name+layer_name[0]+'elemwise')
    gated_data_2=mx.sym.reshape(data=flat_elemwise_2,shape=target_shape, name=name+layer_name[1]+'elemwise')
    
    return gated_data_1,gated_data_2

def residual_gate_broadcast(data,name,layer_name,workspace,gate_prefix,image_shape, gate_act='relu',gate_init=mx.initializer.One(),lr_mult=0.001, wd_mult=0):
    
    if gate_prefix is not None:
        gate_name=name+layer_name+'_'+gate_prefix+'_broadcast_gate'
    else:
        gate_name=name+layer_name+'_broadcast_gate'
    gate = mx.sym.Variable(gate_name, init=gate_init,lr_mult=lr_mult,wd_mult=wd_mult,shape=(1,1,1,1),dtype='float32')
    gate=mx.sym.Activation(data=gate, act_type=gate_act, name=name+layer_name+'gate_act')
    gated_data=mx.sym.broadcast_mul(gate,data)

    
    return gated_data


def residual_unit_2branch(data, num_filter, stride, dim_match,image_shape, name,gate_prefix,coupled,data_a=None,prefix=None,gated=False, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
        
    """
    if prefix is not None:
        name = prefix+name
    if bottle_neck:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        if data_a is not None:
            #print 'I am not none'
            bn1_a = mx.sym.BatchNorm(data=data_a, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='_a_'+name + '_bn1')
            act1_a = mx.sym.Activation(data=bn1_a, act_type='relu', name='_a_'+name + '_relu1')
            conv1_a = mx.sym.Convolution(data=act1_a, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name='_a_'+name + '_conv1')
        else:
            
            bn1_a = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='_a_'+name + '_bn1')
            act1_a = mx.sym.Activation(data=bn1_a, act_type='relu', name='_a_'+name + '_relu1')
            conv1_a = mx.sym.Convolution(data=act1_a, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name='_a_'+name + '_conv1')
        
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
       
        
        bn2_a = mx.sym.BatchNorm(data=conv1_a, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='_a_'+name + '_bn2')
        act2_a = mx.sym.Activation(data=bn2_a, act_type='relu', name='_a_'+name + '_relu2')
        conv2_a = mx.sym.Convolution(data=act2_a, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name='_a_'+name + '_conv2')
        bn3_a = mx.sym.BatchNorm(data=conv2_a, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='_a_'+name + '_bn3')
        act3_a = mx.sym.Activation(data=bn3_a, act_type='relu', name='_a_'+name + '_relu3')
        conv3_a = mx.sym.Convolution(data=act3_a, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name='_a_'+name + '_conv3')
        if gated:
            if coupled:
                conv3,conv3_a = residual_double_gate_coupled(conv3,conv3_a,name,layer_name=['_conv3','_a_conv3'],image_shape=image_shape,gate_prefix=gate_prefix, gate_act='relu',gate_init=mx.initializer.Normal(mu=0.8,sigma=0.01),lr_mult=1, wd_mult=0,workspace=workspace)
            else:
                conv3 = residual_gate(conv3,name,layer_name='_conv3',image_shape=image_shape,gate_prefix=gate_prefix, gate_act='relu',gate_init=mx.initializer.Normal(0.0),lr_mult=1, wd_mult=0,workspace=workspace)
                
            #mx.initializer.One()
                conv3_a = residual_gate(conv3_a,'_a_'+name,layer_name='_conv3',image_shape=image_shape,gate_prefix=gate_prefix, gate_act='relu',gate_init=mx.initializer.Normal(0.5),lr_mult=1, wd_mult=0,workspace=workspace)
           
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + conv3_a + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if gated:
            conv2 = residual_gate(conv2, name,layer_name='_conv2', image_shape=image_shape, gate_act='relu',gate_prefix=gate_prefix,gate_init=mx.initializer.One(),lr_mult=0.001, wd_mult=0,workspace=workspace)
            
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut                                                                  


def residual_unit(data, num_filter, stride, dim_match,image_shape, name,gate_prefix,prefix=None,gated=False, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
        
    """
    if prefix is not None:
        name = prefix+name
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if gated:
            conv3 = residual_gate(conv3,name,layer_name='_conv3',image_shape=image_shape,gate_prefix=gate_prefix, gate_act='relu',gate_init=mx.initializer.One(),lr_mult=1, wd_mult=0,workspace=workspace)
           
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if gated:
            conv2 = residual_gate(conv2, name,layer_name='_conv2', image_shape=image_shape, gate_act='relu',gate_prefix=gate_prefix,gate_init=mx.initializer.One(),lr_mult=0.001, wd_mult=0,workspace=workspace)
            
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut
    
def residual_unit_mbranch(data, num_filter, num_branch, stride, dim_match,image_shape, name,gate_prefix,prefix=None,gated=False, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tupe
        Stride used in convolution
    dim_match : Boolen
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
        
    """
    if prefix is not None:
        name = prefix+name
    out_conv=[]    
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        for b in range(num_branch):
            if b>0:
                local_name= str(b+1)+'_'+name
                gate_init=mx.initializer.Zero()
            else:
                local_name=name
                gate_init=mx.initializer.One()
                
            bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=local_name + '_bn1')
            act1 = mx.sym.Activation(data=bn1, act_type='relu', name=local_name + '_relu1')
            conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                       no_bias=True, workspace=workspace, name=local_name + '_conv1')
            bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=local_name + '_bn2')
            act2 = mx.sym.Activation(data=bn2, act_type='relu', name=local_name + '_relu2')
            conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                       no_bias=True, workspace=workspace, name=local_name + '_conv2')
            bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=local_name + '_bn3')
            act3 = mx.sym.Activation(data=bn3, act_type='relu', name=local_name + '_relu3')
            conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                       workspace=workspace, name=local_name + '_conv3')
            if gated:
                conv3 = residual_gate(conv3,local_name,layer_name='_conv3',image_shape=image_shape,gate_prefix=gate_prefix, gate_act='relu',gate_init=gate_init,lr_mult=0.1, wd_mult=0,workspace=workspace)
            out_conv.append(conv3)
    
           
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return sum(out_conv) + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if gated:
            conv2 = residual_gate(conv2, name,layer_name='_conv2', image_shape=image_shape, gate_act='relu',gate_prefix=gate_prefix,gate_init=mx.initializer.One(),lr_mult=1, wd_mult=0,workspace=workspace)
            
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut    
    
    
    
def resnet(units, num_stages, filter_list, num_classes,rescale_grad,image_shape,dropout=0,bilinear=False,gate_prefix=None,active=None,gated=False,bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of multitask, it accepts [0] entries in the num_classes, allowing not to create all the decision layers based on the bucket key (input of the relative get symbol) but still mantain the output naming convetion --> successive modification will allow to specifiy more differences across buckets by introducing some bucket-scopes for the names
    Parameters
    weights= weightnama,
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : list
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1),gate_prefix=gate_prefix,prefix=None,gated=gated,image_shape=image_shape, bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck,gate_prefix=gate_prefix, prefix=None, gated=gated, image_shape=image_shape, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    pre_shape = relu1.infer_shape(data=(1,image_shape[0],image_shape[1],image_shape[2]))
    print "mio shape" , pre_shape[1]

    if bilinear:
        #bilin1=mx.sym.Convolution(data=relu1, num_filter=4000, kernel=(1,1), stride=(1,1), pad=(0,0),
                                   #no_bias=False, workspace=workspace, name=  'no_sushi_conv_out')
        #bilin1=Had_unit(relu1,4000,8000,'had_',workspace,act_type='tanh')
        bilin1=Bil_unit(relu1,'bilin1',workspace)
        #swapped=mx.sym.SwapAxis(data=bn1,dim1=1,dim2=3)
        
    
        #bilin1=MCB_unit(swapped,1500,'bilinear1',compute_size = 1500, ifftflag = True)
        #swapped2=mx.sym.SwapAxis(data=bilin1,dim1=3,dim2=1)
        bn2 = mx.sym.BatchNorm(data=bilin1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn2')
        relu2 = mx.sym.Activation(data=bn2, act_type='relu', name='relu2')
        pool1 = mx.symbol.Pooling(data=relu2, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
        
    else:
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
        
    flat = mx.symbol.Flatten(data=pool1)
    if dropout>0:
        print "ho dropout %.2f" %(dropout)
        flat = mx.sym.Dropout(flat, p = dropout)
    pool_shape = pool1.infer_shape(data=(1,image_shape[0],image_shape[1],image_shape[2]))
    print "mio shape" , pool_shape[1]
    
    
    
    
    fc_t = []
    fcw_t = []
    s_t = []
    label_names=[]
    task_id=0
    if type(num_classes) is list:
        
        for t in range(len(num_classes)):
            fcw_t.append(mx.sym.Variable(name = 'fc%d_weight' % (t+1),init = mx.initializer.Xavier()))#weight=fcw_t[t]
            fc_t.append(mx.symbol.FullyConnected(data=flat,weight=fcw_t[t], num_hidden=num_classes[t], name='fc%d' % (t+1)))

            if active[t] != 0:
                #print active[t]
                s_t.append(mx.symbol.SoftmaxOutput(data=fc_t[t], name='softmax%d' %(t+1),grad_scale=rescale_grad[t]))
                label_names.append('softmax%d_label'%(t+1))
                #task_id += 1
        out= mx.sym.Group(s_t)
    else:
        fcw=mx.sym.Variable(name = 'fc%d_weight' % (1),init = mx.initializer.Xavier())
        fc=mx.symbol.FullyConnected(data=flat,weight=fcw, num_hidden=num_classes, name='fc%d' % (1))
        out=mx.symbol.SoftmaxOutput(data=fc, name='softmax%d' %(1),grad_scale=rescale_grad)
        label_names.append('softmax%d_label'%(1))
            
    return out,['data'],label_names 

def resnet_2branch(units, num_stages, filter_list, num_classes,rescale_grad,image_shape,coupled,gate_prefix=None,active=None,gated=False,bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of multitask, it accepts [0] entries in the num_classes, allowing not to create all the decision layers based on the bucket key (input of the relative get symbol) but still mantain the output naming convetion --> successive modification will allow to specifiy more differences across buckets by introducing some bucket-scopes for the names
    Parameters
    weights= weightnama,
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : list
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        
        body_a = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="_a_conv0", workspace=workspace)
        body_a = mx.sym.BatchNorm(data=body_a, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='_a_bn0')
        body_a = mx.sym.Activation(data=body_a, act_type='relu', name='_a_relu0')
        body_a = mx.symbol.Pooling(data=body_a, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
        
        
    for i in range(num_stages):
        if i==0:
            #different conv0 for each branch
            body = residual_unit_2branch(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,data_a=body_a, 
                                 name='stage%d_unit%d' % (i + 1, 1),gate_prefix=gate_prefix,coupled=coupled,prefix=None,gated=gated,image_shape=image_shape, bottle_neck=bottle_neck,workspace=workspace,memonger=memonger)
        else:  
            body = residual_unit_2branch(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                                 name='stage%d_unit%d' % (i + 1, 1),gate_prefix=gate_prefix,coupled=coupled,prefix=None,gated=gated,image_shape=image_shape, bottle_neck=bottle_neck, workspace=workspace,memonger=memonger)
        for j in range(units[i]-1):
            
            
            #if i==num_stages-1 and j==units[i]-2:
                
            #else: 
            body = residual_unit_2branch(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                     bottle_neck=bottle_neck,gate_prefix=gate_prefix,coupled=coupled, prefix=None, gated=gated, image_shape=image_shape, workspace=workspace, memonger=memonger)
                
    body = residual_unit(body, filter_list[i+1], (1,1), True, name='final_unit', bottle_neck=bottle_neck,gate_prefix=gate_prefix, prefix=None, gated=False, image_shape=image_shape, workspace=workspace, memonger=memonger)       
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    #here ends the task specific
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc_t = []
    fcw_t = []
    s_t = []
    label_names=[]
    task_id=0
    if type(num_classes) is list:
        
        for t in range(len(num_classes)):
            fcw_t.append(mx.sym.Variable(name = 'fc%d_weight' % (t+1),init = mx.initializer.Xavier()))#weight=fcw_t[t]
            fc_t.append(mx.symbol.FullyConnected(data=flat,weight=fcw_t[t], num_hidden=num_classes[t], name='fc%d' % (t+1)))

            if active[t] != 0:
                #print active[t]
                s_t.append(mx.symbol.SoftmaxOutput(data=fc_t[t], name='softmax%d' %(t+1),grad_scale=rescale_grad[t]))
                label_names.append('softmax%d_label'%(t+1))
                #task_id += 1
        out= mx.sym.Group(s_t)
    else:
        fcw=mx.sym.Variable(name = 'fc%d_weight' % (1),init = mx.initializer.Xavier())
        fc=mx.symbol.FullyConnected(data=flat,weight=fcw, num_hidden=num_classes, name='fc%d' % (1))
        out=mx.symbol.SoftmaxOutput(data=fc, name='softmax%d' %(1),grad_scale=rescale_grad)
        label_names.append('softmax%d_label'%(1))
            
    return out,['data'],label_names  

def resnet_mbranch(units, num_stages,num_branch, filter_list, num_classes,rescale_grad,image_shape,gate_prefix=None,active=None,gated=False,bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of multitask, it accepts [0] entries in the num_classes, allowing not to create all the decision layers based on the bucket key (input of the relative get symbol) but still mantain the output naming convetion --> successive modification will allow to specifiy more differences across buckets by introducing some bucket-scopes for the names
    Parameters
    weights= weightnama,
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : list
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        
        body = residual_unit_mbranch(body, filter_list[i+1],num_branch, (1 if i==0 else 2, 1 if i==0 else 2), False, name='stage%d_unit%d' % (i + 1, 1),gate_prefix=gate_prefix,prefix=None,gated=gated,image_shape=image_shape, bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit_mbranch(body, filter_list[i+1],num_branch, (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2), bottle_neck=bottle_neck,gate_prefix=gate_prefix, prefix=None, gated=gated, image_shape=image_shape, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    #here ends the task specific
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc_t = []
    fcw_t = []
    s_t = []
    label_names=[]
    task_id=0
    if type(num_classes) is list:
        
        for t in range(len(num_classes)):
            fcw_t.append(mx.sym.Variable(name = 'fc%d_weight' % (t+1),init = mx.initializer.Xavier()))#weight=fcw_t[t]
            fc_t.append(mx.symbol.FullyConnected(data=flat,weight=fcw_t[t], num_hidden=num_classes[t], name='fc%d' % (t+1)))

            if active[t] != 0:
                #print active[t]
                s_t.append(mx.symbol.SoftmaxOutput(data=fc_t[t], name='softmax%d' %(t+1),grad_scale=rescale_grad[t]))
                label_names.append('softmax%d_label'%(t+1))
                #task_id += 1
        out= mx.sym.Group(s_t)
    else:
        fcw=mx.sym.Variable(name = 'fc%d_weight' % (1),init = mx.initializer.Xavier())
        fc=mx.symbol.FullyConnected(data=flat,weight=fcw, num_hidden=num_classes, name='fc%d' % (1))
        out=mx.symbol.SoftmaxOutput(data=fc, name='softmax%d' %(1),grad_scale=rescale_grad)
        label_names.append('softmax%d_label'%(1))
            
    return out,['data'],label_names  


def get_symbol(num_classes,active,rescale_grad, num_layers, image_shape,dropout=0,bilinear=False,gate_prefix=None,gated=False, conv_workspace=256, **kwargs):
    """
    This can be used in a bucketing scenario where the bucket key is a list of num_classes [...], might generate errors if the same index has different >0 values at differnt buckets.
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  active      = active,
                  rescale_grad = rescale_grad,
                  gated = gated,
                  dropout=dropout,
                  bilinear = bilinear,
                  gate_prefix=gate_prefix,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)

def get_symbol_2branch(num_classes,active,rescale_grad, num_layers, image_shape,coupled=False,gate_prefix=None,gated=False, conv_workspace=256, **kwargs):
    """
    This can be used in a bucketing scenario where the bucket key is a list of num_classes [...], might generate errors if the same index has different >0 values at differnt buckets.
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet_2branch(units = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  active      = active,
                  rescale_grad = rescale_grad,
                  gated = gated,
                  coupled=coupled,        
                  gate_prefix=gate_prefix,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)


def get_symbol_mbranch(num_classes,active,rescale_grad,num_branch, num_layers, image_shape,gate_prefix=None,gated=False, conv_workspace=256, **kwargs):
    """
    This can be used in a bucketing scenario where the bucket key is a list of num_classes [...], might generate errors if the same index has different >0 values at differnt buckets.
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))

    return resnet_mbranch(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  num_branch=2,        
                  active      = active,
                  rescale_grad = rescale_grad,
                  gated = gated,
                  gate_prefix=gate_prefix,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)


def get_multi_symbol(num_classes,active,rescale_grad, num_layers, image_shape,gate_prefix=None,gated=False, conv_workspace=256, **kwargs):
    """
    This can be used in a bucketing scenario where the bucket key is a list of num_classes [...], might generate errors if the same index has different >0 values at differnt buckets.
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it youself".format(num_layers))
        out=[]    
        for i in range(len(gate_prefix)):
            
            sym,data,label=(resnet(units       = units,
                      num_stages  = num_stages,
                      filter_list = filter_list,
                      num_classes = num_classes,
                      active      = active[i],
                      rescale_grad = rescale_grad,
                      gated = gated,
                      prefix=str(i),             
                      gate_prefix=gate_prefix[i],
                      image_shape = image_shape,
                      bottle_neck = bottle_neck,
                      workspace   = conv_workspace))
            out.append(sym)
        return mx.sym.Group(out),data,label 
    
def hyper_resnet(units, num_stages, filter_list, num_classes,active,rescale_grad,image_shape,gated=True, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of multitask, it accepts [0] entries in the num_classes, allowing not to create all the decision layers based on the bucket key (input of the relative get symbol) but still mantain the output naming convetion --> successive modification will allow to specifiy more differences across buckets by introducing some bucket-scopes for the names
    Parameters
    weights= weightnama,
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : list
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')
    #first the hypernet body
    for i in range(1):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1),prefix='hyper_',gated=False,image_shape=image_shape, bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck,prefix='hyper_',gated=False,image_shape=image_shape, workspace=workspace, memonger=memonger)
    
    #now the gated resnet body + the hyper fc layers   
    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1),prefix=None,gated=gated,image_shape=image_shape, bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck,prefix=None,gated=gated,image_shape=image_shape, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    #here ends the task specific
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc_t = []
    fcw_t = []
    s_t = []
    label_names=[]
    task_id=0
    if type(num_classes) is list:
        for t in range(len(num_classes)):
            fcw_t.append(mx.sym.Variable(name = 'fc%d_weight' % (t+1),init = mx.initializer.Xavier()))#weight=fcw_t[t]
            fc_t.append(mx.symbol.FullyConnected(data=flat,weight=fcw_t[t], num_hidden=num_classes[t], name='fc%d' % (t+1)))

            if active[t] != 0:
                s_t.append(mx.symbol.SoftmaxOutput(data=fc_t[t], name='softmax%d' %(t+1),grad_scale=rescale_grad))
                label_names.append('softmax%d_label'%(t+1))
                #task_id += 1
        out= mx.sym.Group(s_t)
    else:
        fcw=mx.sym.Variable(name = 'fc%d_weight' % (1),init = mx.initializer.Xavier())
        fc=mx.symbol.FullyConnected(data=flat,weight=fcw, num_hidden=num_classes, name='fc%d' % (1))
        out=mx.symbol.SoftmaxOutput(data=fc, name='softmax%d' %(1),grad_scale=rescale_grad)
        label_names.append('softmax%d_label'%(1))
            
    return out,['data'],label_names  
