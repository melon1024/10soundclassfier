# -*- coding: utf-8 -*-
import numpy as np

tc = np.ndarray(shape=[0,39], dtype=np.float32)
td = np.ndarray(shape=[0,10], dtype=np.float32)
td_num = np.ndarray(shape=[0], dtype=np.float32)

for i in range(10):
    # 파일에서 X를 가져온다.
    filename = "./"+str(i)+".npz"
    data = np.load(filename)
    x = data['X']
    data.close()
    # X의 갯수에 맞는 적절한 정답 Y를 만든다.
    ans = [0,0,0,0,0,0,0,0,0,0]
    ans[i] = 1
    y = []
    y_num = []
    for j in range(x.shape[0]):
        y.append(ans)
        y_num.append(i)
    y = np.array(y)
    y_num = np.array(y_num)
    # tc, td 에 데이터를 모은다.
    tc = np.append(tc, x, axis=0)
    td = np.append(td, y, axis=0)
    td_num = np.append(td_num, y_num, axis=0) 
    print(filename,x.shape,"elements are saved.")

# 데이터가 모두 모아졌으므로, 파일로 저장한다.
tc = tc.astype(np.float32)
td = td.astype(np.float32)
td_num = td_num.astype(np.float32)


np.savez("mytotaldata", X=tc, Y=td, Y_num=td_num)

