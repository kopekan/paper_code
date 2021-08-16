#evaluation function written in Lattice
import numpy as np
def reverse_style(input_string):
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string

def get_ner_BIO(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    
    list_len = len(label_list)
    begin_label = 'B-'
    inside_label = 'I-' 
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag == '':
                whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
                index_tag = current_label.replace(begin_label,"",1)
            else:
                tag_list.append(whole_tag + ',' + str(i-1))
                whole_tag = current_label.replace(begin_label,"",1)  + '[' + str(i)
                index_tag = current_label.replace(begin_label,"",1)

        elif inside_label in current_label:
            if current_label.replace(inside_label,"",1) == index_tag:
                whole_tag = whole_tag 
            else:
                if (whole_tag != '')&(index_tag != ''):
                    tag_list.append(whole_tag +',' + str(i-1))
                whole_tag = ''
                index_tag = ''
        else:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix

def get_ner_BIOES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B-'
    end_label = 'E-'
    single_label = 'S-'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        current_label = label_list[i].upper()
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)
            
        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag +',' + str(i))
            whole_tag = ''
            index_tag = ''
        else:
            continue
    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    # print stand_matrix
    return stand_matrix


def get_ner_fmeasure(return_result_table,golden_lists, predict_lists, myidx2label=None, label_type="BIOES"):
    print('label type:', label_type)
    sent_num = len(golden_lists)
    right_num = 0
    golden_num = 0
    predict_num = 0
    right_tag = 0
    all_tag = 0
    alltype_f = []
    #to every type of tag
    import pandas as pd
    colnames = []
    result_count = {}
    if myidx2label==None:
        myidx2label = idx2label
    for i in myidx2label.values():
        if len(i)>2 and i[2:] not in colnames:
            colnames.append(i[2:])
    colnames.append('total')
    for i in colnames:
        result_count[i] = {'true':0,'predict':0,'answer':0}
    result_table = pd.DataFrame(0.0,index=colnames, columns = ['P', 'R', 'F'])
    
    for idx in range(sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx]
        list_len = 0            
        for idy in range(min(len(golden_list), 128)): #限制原始句子長度<=128
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BIOES":
            gold_matrix = get_ner_BIOES(golden_list)
            pred_matrix = get_ner_BIOES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_num += len(gold_matrix)
        predict_num += len(pred_matrix)
        right_num += len(right_ner)
        if return_result_table:
            for i in right_ner:
                for j in colnames:
                    if j.upper() in i:
                        result_count[j]['true']+=1
            for i in gold_matrix:
                for j in colnames:
                    if j.upper() in i:
                        result_count[j]['answer']+=1
            for i in pred_matrix:
                for j in colnames:
                    if j.upper() in i:
                        result_count[j]['predict']+=1
    if return_result_table:
        for i in colnames:
            if result_count[i]['predict']>0:
                result_table['P'][i] = round(result_count[i]['true']/result_count[i]['predict'],4)
            if result_count[i]['answer']>0:
                result_table['R'][i] = round(result_count[i]['true']/result_count[i]['answer'],4)
            if result_table['P'][i]+result_table['R'][i]>0:
                result_table['F'][i] = round(2*result_table['P'][i]*result_table['R'][i]/(result_table['P'][i]+result_table['R'][i]),4)
        result_count['total']['true']=right_num
        result_count['total']['predict']=predict_num
        result_count['total']['answer']=golden_num
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    #accuracy is for tag accuracy, not for entity accuracy
    accuracy = (right_tag+0.0)/all_tag 
    print ("Accuracy: ", right_tag,"/",all_tag,"=",accuracy)
    print ("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    result_table['P']['total'], result_table['R']['total'], result_table['F']['total'] = round(precision,2), round(recall,2), round(f_measure,2)
    return result_count, precision, recall, f_measure, result_table

def transfer_idx2label(data, idx2label):
    return [[idx2label[j] for j in i] for i in data]

def evaluation(pred, ans, return_result_table=True, pred_idx2label=None, ans_idx2label = None):
    if pred_idx2label!=None and ans_idx2label==None:
        ans_idx2label = pred_idx2label
    elif pred_idx2label==None and ans_idx2label!=None:
        pred_idx2label = ans_idx2label
    elif pred_idx2label==None and ans_idx2label==None:
        print('please give the idx2label!')
        return 0, 0, 0, 0
#         print('warning! 未設定predict & answer idx2label')
    predict = transfer_idx2label(pred, pred_idx2label)
    answer = transfer_idx2label(ans, ans_idx2label)
#    print('shape of predict:{}, ans:{}'.format(np.shape(pred), np.shape(ans)))
    if np.shape(pred)!=np.shape(ans):
        print('Alert! predict and answer size are different!')
        print('shape of predict:{}, ans:{}'.format(np.shape(pred), np.shape(ans)))
        print('instance shape of predict:', np.shape(pred[0]))
        print('instance shape of answer :', np.shape(ans[0]))
    result_count, p, r, f, result_table = get_ner_fmeasure(return_result_table,answer, predict, ans_idx2label)
    return p,r,f, result_table, result_count
