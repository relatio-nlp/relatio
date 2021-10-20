import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Load dataset 
sentiment_data_raw = pd.read_csv("sentiment_narratives.csv")

# We want a scatter plot in 1-D, as matplotlib must take (x,y) as argument, we add a fake dimension of y = [0,0,0,0,0...]
xcoords = sentiment_data_raw["sentiment_compound"]
ycoords = np.zeros(len(sentiment_data_raw))

# Shared x-axis for multiple ranges of data (+ve sentiment and -ve sentiment are separate)
fig,(ax, ax2) = plt.subplots(1, 2, sharey=True, figsize=(15,15))

# -ve sentiment is red, positive is green!
ax.scatter(xcoords, ycoords, color='red', label = "negative")
ax2.scatter(xcoords, ycoords, color='green', label = "positive")

ax.set_xlim(-0.95, -0.55) 
ax2.set_xlim(0.55, 1.3)

ax.spines.top.set_visible(False)
ax2.spines.top.set_visible(False)

ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax.yaxis.tick_left()
ax2.yaxis.tick_right()

# don't put tick labels at the top
#ax.tick_params(labeltop='off')
#ax2.tick_params(labeltop='off') 


'''
    There is no generic way to plot this data in the way we want, as the labels have arbitrary sizes and distributions. I use some base logic,
    where I distinguish between indices which have composite divisors vs prime. For composite, I start with the highest possible multiple in the
    set, and reduce, and fix the Y coordinate of the labels, based on the size and relative distance. If needed, I adjust the x-coordinate of the 
    labels later. Then for the remaining prime numbers, I alternate between odd and even indices and place them alternately. Indices 0 and 1 are 
    treated separately. The code is a little hacky but it generates a good plot, making a general way for this plot is hard due to the nature of 
    the dataset.
'''

# Remaining indices in the dataset, after the main conditions have been met.
remaining = []

for i in range(len(sentiment_data_raw)):

    # Get the annotations 
    annotation_text = sentiment_data_raw["narrative"][i]
    act_axis = ax if xcoords[i] < 0 else ax2

    # Split long annotations into multiple lines
    # If length is less than 4, don't split.
    # If length is 4, split as 2 + 2
    # If length is >4, split as 3 + 2(3)
    if len(annotation_text.split(" "))==4:
        list_of_words = annotation_text.split(" ")
        annotation_text = ' '.join(word for word in list_of_words[0:2]) +"\n" + ' '.join(word for word in list_of_words[2:])

    if len(annotation_text.split(" "))>=5:
        list_of_words = annotation_text.split(" ")
        annotation_text = ' '.join(word for word in list_of_words[0:3]) +"\n" + ' '.join(word for word in list_of_words[3:])

    # Plot the labels for all composite indices
    if i!=0 and i!=1:

        if i%18==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]+0.03, ycoords[i]+0.035), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        # Special case of prime index for better formattting
        elif i%17==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]+0.095, ycoords[i]-0.045), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%16==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i], ycoords[i]-0.03), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%15==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]-0.01, ycoords[i]+0.02), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%12==0 :
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i], ycoords[i]-0.04), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%9==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]+0.07, ycoords[i]-0.03), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%6==0 :
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]-0.1, ycoords[i]-0.04), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%3==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]+0.05, ycoords[i]+0.025), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%10==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]+0.01, ycoords[i]-0.03), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%8==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i], ycoords[i]+0.01), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%4==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]-0.2, ycoords[i]+0.03), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        elif i%2==0:
            act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (xcoords[i]+0.02, ycoords[i]+0.005), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        # Store the prime number indices    
        else:
            remaining.append(i)
        
        remaining = sorted(remaining)
        
    # Handle the case for 0 and 1 as the index
    elif i==0:
        act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (0.5, ycoords[i]-0.01), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    else:
        act_axis.annotate(annotation_text, (xcoords[i], ycoords[i]), (0.65, ycoords[i]-0.03), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

print(remaining)

# Plot labels for the prime number indices    
for j in remaining:

    annotation_text = sentiment_data_raw["narrative"][j]
    act_axis = ax if xcoords[j] < 0 else ax2

    if len(annotation_text.split(" "))==4:
        list_of_words = annotation_text.split(" ")
        annotation_text = ' '.join(word for word in list_of_words[0:2]) +"\n" + ' '.join(word for word in list_of_words[2:])

    if len(annotation_text.split(" "))>=5:
        list_of_words = annotation_text.split(" ")
        annotation_text = ' '.join(word for word in list_of_words[0:3]) +"\n" + ' '.join(word for word in list_of_words[3:])

    # Alternate between odd and even for the prime number indices
    if remaining.index(j)%2==0:
        act_axis.annotate(annotation_text, (xcoords[j], ycoords[j]), (xcoords[j]+0.00, ycoords[j]+0.045), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    else:
        act_axis.annotate(annotation_text, (xcoords[j], ycoords[j]), (xcoords[j], ycoords[j]-0.05), arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

#ax.legend(loc='lower right')
#ax2.legend(loc='lower right')
d = .010

kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d,1+d),(-d,+d), **kwargs) # top-left diagonal
kwargs.update(transform=ax2.transAxes) # switch to the bottom axes
ax2.plot((-d,d),(-d,+d), **kwargs) # top-right diagonal


plt.show()
fig.tight_layout()
fig.savefig('sentiment_narratives.png')



