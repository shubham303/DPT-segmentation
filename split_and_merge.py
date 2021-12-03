import numpy as np


def  connected_component(img):
	H, W = img.shape
	out = np.zeros((H,W))
	visited=np.zeros((H,W))
	label=1
	for i in range(0,H):
		for j in range(0, W):
			if out[i,j]==0:
				que=[]
				que.append((i,j))
				visited[i, j] = 1
				while len(que) > 0:
					a, b = que.pop(0)
					
					out[a,b] = label
					if a<H-1 and img[a+1, b] == img[a, b] and not visited[a+1, b]:
						que.append((a+1, b))
						visited[a + 1, b]=1
					if a>0 and img[a-1, b] == img[a, b]and not visited[a-1, b]:
						que.append((a-1, b))
						visited[a - 1, b]=1
					if b<W-1 and img[a, b+1] == img[a, b]and not visited[a, b+1]:
						que.append((a, b+1))
						visited[a, b + 1]=1
					if b>0 and img[a, b-1] == img[a, b]and not visited[a, b-1]:
						que.append((a, b-1))
						visited[a, b - 1]=1
				label+=1
	return out