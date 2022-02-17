#演示内容：利用tensorflow实现的循环神经网络RNN（本程序使用了LSTM）来做语言模型，并输出其困惑度。
#语言模型主要是根据一段给定的文本来预测下一个词最有可能是什么。困惑度用于评价语言模型。困惑度越小，则模型的性能越好。
 
import tensorflow as tf
 
#reader模块用于读取数据集，以及做训练集、验证集和测试集的切分等。如果无法导入reader模块，也可以从网上下载reader.py，加入到同级目录即可，下载地址：https://github.com/tensorflow/models/blob/master/tutorials/rnn/ptb/reader.py
import reader 
import numpy as np
 
DATA_PATH='data' #data目录中应该预先存放本程序中要用到的语料，即PTB(Penn Treebank Dataset)数据集，也就是宾州树库数据集 。PTB数据集是语言模型研究常用的数据集。其下载地址在 Tomas Mikolov的主页： https://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz。解压后，将其中的整个data文件夹拷贝到当前目录即可。数据集共有9998个单词，加上稀有词语的特殊符号<unk>和语句的结束标记，共有10000个单词。
 
#超参数的设置
HIDDEN_SIZE=200#size of RNN hidden state，指表示每个单词的词向量的维度
NUM_LAYERS=2   #表示使用了两个LSTM层
VOCAB_SIZE=10000 #词汇表中词的个数
 
LEARNING_RATE=1.0 # 设置学习速率超参数
TRAIN_BATCH_SIZE=20  # 训练阶段每个数据批量设置为多少个样本。在本例中每个样本是指由若干个词构成的一个序列
TRAIN_NUM_STEP=35    # 训练阶段时间步的个数，即在训练阶段中文本数据的截断长度，也可以称为序列长度seq_length
 
# 可以将测试数据看成一个很长的序列，所以下面这两个参数都设置为1
EVAL_BATCH_SIZE=1    #在后面main函数中计算valid_batch_len时会用到
EVAL_NUM_STEP=1      #在后面main函数中计算valid_epoch_size时会用到
NUM_EPOCH=2    #训练的轮数
KEEP_PROB=0.5  # 节点不被dropout的概率
MAX_GRAD_NORM=5 #这个超参数后面会用到，用于避免梯度膨胀问题
 
# 类PTBModel的定义，是这个多层循环神经网络模型的描述
class PTBModel(object):
      #初始化函数，其参数包括当前是否在训练阶段，批量的大小，数据的截断长度即序列长度等
    def __init__(self,is_training,batch_size,num_steps): 
        # 根据给定的参数数值来设置使用的batch大小和时间步数（截断长度）
        self.batch_size=batch_size
        self.num_steps=num_steps
         
        # 定义输入层的数据维度为: batch_size * num_steps。 这里tf.placeholder可以理解为采用占位符进行占位，等以后实际运行时进行具体填充。等建立session，运行模型的时候才feed数据        
        self.input_data=tf.placeholder(tf.int32,[batch_size,num_steps])
         
        # 定义输出层的数据维度为：batch_size*num_steps。num_steps相当于seq_length
        self.targets=tf.placeholder(tf.int32,[batch_size,num_steps])
         
        # 定义使用LSTM结构作为循环体的基本结构。每个单词向量的维度为HIDDEN_SIZE
        lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        #如果是在训练阶段，则使用dropout.此时每个单元以（1-keep_prob）的概率不工作，目的是防止过拟合。
        if is_training:
            lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell,output_keep_prob=KEEP_PROB)
         
        #将多层RNN单元封装到一个单元cell中。层的个数NUM_LAYERS前面已经设定为2了
        cell=tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*NUM_LAYERS) 
                 
        # 使用zero_state函数来初始化网络的状态
        self.initial_state=cell.zero_state(batch_size,tf.float32)
         
        # 下面两条语句是将单词转换成向量
        #VOCAL_SIZE即单词的总数；  HIDDEN_SIZE，即每个单词的向量维度。所以 embedding参数的维度为VOCAB_SIZE * HIDDEN_SIZE. #则inputs的维度为：batch_size * num_steps * HIDDEN_SIZE    
        embedding=tf.get_variable('embedding',[VOCAB_SIZE,HIDDEN_SIZE])       
        inputs=tf.nn.embedding_lookup(embedding,self.input_data)
         
        # 如果在训练阶段，则进行dropout
        if is_training:
            inputs=tf.nn.dropout(inputs,KEEP_PROB)
             
        #定义LSTM结构的输出列表
        outputs=[]
        # state存储LSTM的初始状态
        state=self.initial_state
        #TensorFlow提供了Variable Scope 机制，用于共享变量
        with tf.variable_scope('RNN'): 
            #对于每个时间步
            for time_step in range(num_steps):
                if time_step>0:
                    tf.get_variable_scope().reuse_variables() #重用变量
                 
                # 从输入数据inputs中获取当前时刻的输入并传入LSTM的单元，并得到输出。
                #每次的输出都是一个张量，其shape=(20, 200)，其中20是BATCH_SIZE，200是词向量维度
                cell_output,state=cell(inputs[:,time_step,:],state)
               #将当前单元的输出加入到输出的列表中
                outputs.append(cell_output)
         
        # 将输出的列表利用tf.concat函数变成（batch,hidden_size*num_steps）的形状，然后再reshape成（batch*num_steps,hidden_size）的形状
        #,即为 （20*35， 200）=（700, 200）
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
         
        #下面三条语句利用刚才LSTM的输出向量output乘以权重weight再加上偏置bias，得到最后的预测结果logits
        weight=tf.get_variable('weight',[HIDDEN_SIZE,VOCAB_SIZE])  #weight的形状为 [200, 10000] 
        bias=tf.get_variable('bias',[VOCAB_SIZE])
        logits=tf.matmul(output,weight)+bias
 
        #计算交叉熵损失
        loss=tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits], #logits是刚才预测的结果  shape=(700, 10000)
            [tf.reshape(self.targets,[-1])], # self.targets是正确的结果.这里对其shape进行了调整，变为shape=(700,)
            [tf.ones([batch_size*num_steps],dtype=tf.float32)]# 设置损失的权重。所有的权重都设为1，即不同batch和不同时刻的权重相同
            )
         
        # 计算每个batch的平均损失，并更新网络的状态
        self.cost=tf.reduce_sum(loss)/batch_size
        self.final_state=state
         
        # 如果当前是在训练阶段，则继续进行下面的反向传播以更新梯度
        if not is_training:
            return 
        #返回的是需要训练的变量列表
        trainable_variables=tf.trainable_variables()
         
        # 通过clip_by_global_norm函数控制梯度的大小，避免梯度膨胀问题
        grads,_=tf.clip_by_global_norm(
            tf.gradients(self.cost,trainable_variables),MAX_GRAD_NORM)
         
        #定义模型的优化方法
        optimizer=tf.train.GradientDescentOptimizer(LEARNING_RATE)
        #应用梯度对trainable_variables进行更新
        self.train_op=optimizer.apply_gradients(zip(grads,trainable_variables))
         
         
# 定义的函数run_epoch。使用刚才定义的模型model在数据data上运行train_op并返回perplexity值
def run_epoch(session,model,data,train_op,output_log,epoch_size):
    # 这两个变量是用于计算perplexity的辅助中间变量
    total_costs=0.0
    iters=0
    #对模型进行初始化操作
    state=session.run(model.initial_state)
    # 对每一轮，使用当前数据训练模型
    for step in range(epoch_size):    
          
        #这个run语句会返回并得到预测值y        
        x, y = session.run(data) 
         
        #下面的run语句计算得到cost，即交叉熵值
        cost,state,_=session.run([model.cost, model.final_state, train_op],
                                {model.input_data:x, model.targets:y, model.initial_state:state})
               
        #【然后求和】
        total_costs+=cost
        iters+=model.num_steps
 
        # 如果output_log为真，且每隔100步，就会输出困惑度的数值。困惑度ppx=exp(loss/N)
        if output_log and step % 100 ==0:
            print('After {} steps,perplexity is {}'.format(step,np.exp(total_costs/iters)))   
 
    # 返回perplexity值
    return np.exp(total_costs/iters)
 
#主函数的定义
def main(_):
    # 利用reader.ptb_raw_data 函数 从数据目录DATA_PATH中读取训练数据集，验证数据集和测试数据集
    train_data,valid_data,test_data,_=reader.ptb_raw_data(DATA_PATH)
     
     #得到训练阶段中每个epoch所需要训练的次数，即 train_epoch_size。
    train_data_len = len(train_data)  #得到训练数据的文本长度
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE  #得到训练数据共计需要多少个batch
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP  #再除以时间步的个数，从而得到每个epoch所需要训练的次数
 
    #下同
    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE  
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP
 
    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP
     
    #利用tf.random_uniform_initializer生成均匀分布的随机数
    initializer=tf.random_uniform_initializer(-0.05,0.05)
     
    #定义训练阶段使用的模型，PTBModel是之前定义过的类，其定义了网络结构。
    with tf.variable_scope('language_model',reuse=None,initializer=initializer):  
        train_model=PTBModel(True,TRAIN_BATCH_SIZE,TRAIN_NUM_STEP)
         
    #定义测试阶段使用的模型
    with tf.variable_scope('language_model',reuse=True,initializer=initializer):
        eval_model=PTBModel(False,EVAL_BATCH_SIZE,EVAL_NUM_STEP)
     
    #使用with tf.Session()语句来创建上下文（Context）并执行
    with tf.Session() as sess:
        #初始化变量
        tf.global_variables_initializer().run()
         
        #利用reader.ptb_producer函数切分生成训练集、验证集和测试集上各自的队列
         
        train_queue = reader.ptb_producer(train_data, train_model.batch_size, train_model.num_steps)
        eval_queue = reader.ptb_producer(valid_data, eval_model.batch_size, eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size, eval_model.num_steps)
 
        #使用tf.train.Coordinator()来创建一个线程管理器对象。然后调用tf.train.start_queue_runners，把数据推入内存序列中供计算单元调用
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
         
        # 对于每个epoch进行训练。NUM_EPOCH表示总的训练的轮数
        for i in range(NUM_EPOCH):
            print('In EPOCH: {}'.format(i+1)) #【对于每个epoch】
            # 调用之前的run_epoch函数来训练模型，当然要提供训练的模型，训练数据等
            run_epoch(sess,train_model,train_queue,train_model.train_op,True,train_epoch_size)
             
            #使用验证数据来计算模型的困惑度，并打印输出
            valid_perplexity=run_epoch(sess,eval_model,eval_queue,tf.no_op(),False,valid_epoch_size)
            print('Epoch: {} Validation Perplexity: {}'.format(i+1,valid_perplexity))
             
        # 计算并返回在测试数据集上的困惑度，然后打印输出
        test_perplexity=run_epoch(sess,eval_model,test_queue,tf.no_op(),False,test_epoch_size)
        print('Test Perplexity: {}'.format(test_perplexity))
        
        
       #使用coord.request_stop()来发出终止所有线程的命令，使用coord.join(threads)把线程加入主线程，等待threads结束。
        coord.request_stop()
        coord.join(threads)
 
#主函数的入口
 
if __name__ == '__main__':
    tf.app.run()