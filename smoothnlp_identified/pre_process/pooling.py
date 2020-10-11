import jieba
import time
import logging
import Pool

# 先试试这个函数看跑步跑得了，再试试进程池pool对象定义一定放在main函数下，如果不放在这里会出现如下报错


def multi_processing_data(files_path_feeder, save_path, batch_size=5000):
    """
    multi-processing, execute prepare_data(), save output into hdf5
    files_path_feeder: a generator return files path by batch
    save_path: hdf5 file path like ' ./output.hdf5'
    file_count: total files to be prepare
    batch_size: how many files to be prepared once
    """
    file_cout = 0
    file_list = []
    # 初始化jieba词典，放在后台运行
    jieba.load_userdict('data/dictionary/web_dictionary.txt')
    for root, dirs, files in os.walk(files_path_feeder):
        file_cout = len(files)
        file_list.append(files)

    ck_num = int(file_cout / batch_size)
    iter_times = 0
    rows = 0
    illegal_files = 0

    start_p = time.time()
    logging.info('start prepare_data')
    logging.info('-------------------------------------------------------------')
    for files in files_path_feeder:
        start_l = time.time()
        pool = Pool(45)  # 最大进程数
        output = pool.map(blog_corpus, files)
        pool.close()
        pool.join()

        illegal_files += len([n for n in output if n[0] == -1])  # count illegal files
        output = [n for n in output if n[0] != -1]              # drop illegal_file
        for n in output:                                        # count rows of corpus
            rows += n[1].shape[0]

        # write into hdf5

        # monitor processing
        percentage = (iter_times + 1) / (ck_num + 1)
        done = int(100 * percentage)
        undone = 100 - done
        iter_times += 1
        logging.info('iteration %d th, multi-processing time: %0.2f s' % (iter_times, time.time() - start_l))
        logging.info(''.join(['#'] * done + ['.'] * undone) + (' %0.2f' % (percentage * 100)) + '%')

    corpus.to_csv(save_path, header=False, index=False, seq=',')

    logging.info('-------------------------------------------------------------')
    logging.info('total files %d , illegal %d, effective %d (%0.2f) ' % (
        file_count, illegal_files, file_count - illegal_files,
        (file_count - illegal_files) / file_count))
    logging.info('total rows %d , each row contains 10000 word(coding utf-8)' % (rows))
    logging.info('done prepare_data, processing time: %0.2f s' % (time.time() - start_p))
