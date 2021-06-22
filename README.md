# PredictHandWritten
Construct a classifier to predict handwritten digits using images.  We use `sklean`n package's `digits`  data set. Some of the following helps your to understand the data. 

```
from sklearn.datasets import load_digits
import numpy as np
digits = load_digits()

print('The number of images is: '+ str(digits.data.shape[0]))
print('The number of pixels in an image is: '+ str(digits.data.shape[1]))
print('The targets/labels are: '+ str(np.unique(digits.target)))

import matplotlib.pyplot as plt 
plt.gray() 
plt.matshow(digits.images[100]) #Show the 101th image in the data set.
plt.show()

//The number of images is: 1797
//The number of pixels in an image is: 64
//The targets/labels are: [0 1 2 3 4 5 6 7 8 9]
```

![image](https://user-images.githubusercontent.com/74609915/122909181-44576500-d398-11eb-8a69-25083016c659.png)


```
# Show all 10 digits
for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.imshow(digits.images[i], cmap=plt.cm.gray, vmax=16, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.show()
```

![image](https://user-images.githubusercontent.com/74609915/122909213-4f11fa00-d398-11eb-9b29-a1c2fc91b3fa.png)


The data is in digits.data and the labels are in digits.target. Images have been "squashed" to vectors and stored in digits.data. Construct a classifier to able to predict digits for a given new image. Please refer to the demonstration code blocks above if you have difficulties.
