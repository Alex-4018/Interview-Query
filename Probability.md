#### 1. Running Dog
##### A man and a dog stand at opposite ends of a football field that is 100 feet long. Both start running towards each other.Let's say that the man runs at X ft/s and the dog runs at twice the speed of the man. Each time the dog reaches the man, the dog runs back to the end of the field where it started and then back to the man and repeat.What is the total distance the dog covers once the man finally reaches the end of the field?
<pre>
200 ft : 2 * 200 * (1⁄3+1⁄9+1⁄27+1⁄81+……) =200
</pre>

The man traveled 100 feet, and the dog travels twice as fast, so the dog must have traveled 200 feet in the same time period (their start and end times are the same).

#### 2. Lazy Raters
##### Out of all of the raters, 80% of the raters carefully rate movies and rate 60% of the movies as good and 40% as bad. The other 20% are lazy raters and rate 100% of the movies as good. Assumming all raters rate the same amount of movies, what is the probability that a movie is rated good?
<pre>
P(movie rated good ) = P(non-lazy raters) x P(good rating) + P(lazy raters) x P(good rating)<br> 
                     = (0.8) x (0.6) + (0.2) x (1.0) = 0.68
</pre>

#### 3. Profit-Maximizing Dice Game
##### You're playing casino dice game. You roll a die once. If you reroll, you earn the amount equal to the number on your second roll otherwise, you earn the amount equal to the number on your first roll. Assuming you adopt a profit-maximizing strategy, what would be the expected amount of money you would win?
<pre>
E(Max Profit with two roll chance)= P(roll 4,5,6 at first time and give up second chance) x E [4,5,6] + P(roll 1,2,3 at first time and roll the second chance) x E [1,2,3,4,5,6] 
                                  = 1/2 x 5 + 1/2 x 3.5 = 4.25
</pre>                            
Since the expectation for 1 roll is 3.5. If you get any number large than the expectation, you can stop and take the profit. Otherwise, you choose the second roll and get 3.5 in average.

#### 4. HHT or HTT
##### You're given a fair coin. You flip the coin until either Heads Heads Tails (HHT) or Heads Tails Tails (HTT) appears. Is one more likely to appear first? If so, which one and with what probability?
<pre>
HT appears in both. 
Given that you have an HT, there’s a 50% chance of an H preceding it resulting in a HHT win, 
while for HTT to win you need a T as both a prefix and a suffix T - HT - T so that 1⁄2 * 1⁄2=25%. 
So the odds are 2⁄1 in favor of HHT, or 66% chance.
</pre>

#### 5. Flipping 576 times
##### You flip a fair coin 576 times. Without using a calculator, calculate the probability of flipping at least 312 heads.

<pre>
Use Normal approximation Variance = np(1-p)=144 
                         StdDev   = sqrt(Var)=12
312 = 576⁄2 + 2* StdDev 
Right tail area is 2.2%
</pre>

#### 6. Secret Wins
##### There are 100 students that are playing a coin-tossing game. The students are given a coin to toss. If a student tosses the coin and it turns up heads, they win. If it comes up tails, then they must flip again. If the coin comes up heads the second time, the students will lie and say they have won when they didn't. If it comes up tails then they will say they have lost. If 30 students at the end say they won, how many students actually did win the game?
<pre>
We have to assume that the coin is unfair or the probability of a coin toss to turn out heads/tails is unknown.
p+(1-p)*p =0.3
Solving for p gives, about 16 people in 100 people win the first time, the other 14 lie to say won.
</pre>

#### 7. 500 Cards
##### Imagine a deck of 500 cards numbered from 1 to 500. If all the cards are shuffled randomly and you are asked to pick three cards, one at a time, what's the probability of each subsequent card being larger than the previous drawn card?
<pre>
1/6. It doesn’t matter which 3 cards. There’s 3! ways of ordering the cards, and only one way of ordering them in increasing order.
</pre>

#### 8. Impression Reach
##### Let's say we have a very naive advertising platform. Given an audience of size A and an impression size of B, each user in the audience is given the same random chance of seeing an impression. 
##### 1. Compute the probability that a user sees exactly 0 impressions.  
##### 2. What's the expected value of each person receiving at least 1 impression?
<pre>
a. If the impressions can be repetitive: 
Assume a user can receive many impressions, then the number of a times a user receive an impression follows the binomial distribution (B, 1/A), 
because the total number of impressions is B and the probability a user being selected for an impression is the same for all the audience, i.e., 1/A. 
            For #1 is (1-1/A)^B.
            For #2 is [1-(1-1/A)^B]^A.
b. If the impressions are non-repetitive:
            P(0)=1-B/A
            P(1)= B/A
</pre>

#### 9. Raining in Seattle
##### You are about to get on a plane to Seattle. You want to know if you should bring an umbrella. You call 3 random friends of yours who live there and ask each independently if it's raining. Each of your friends has a 2/3 chance of telling you the truth and a 1/3 chance of messing with you by lying. All 3 friends tell you that "Yes" it is raining.
<pre>
You do need your prior to answer the original question. 
Let A="they all say it's raining" and B="it rains" 
then P(A|B)=8/27 and P(A|not B)=1/27 so (cancelling factors of 27)
P(B|A)=8xP(B)/ (8xP(B)+1x(1−P(B)) = 8P(B)/(1+7P(B))
</pre>

#### 10. First to Six
##### Amy and Brad take turns in rolling a fair six-sided die. Whoever rolls a "6" first wins the game. Amy starts by rolling first. What's the probability that Amy wins?
<pre>
P(amy) = 1⁄6 + 5⁄6*5⁄6*P(amy) => P(amy) = (1/6)/(11/36) = 6⁄11
P(amy) = 1⁄6 + 5⁄6*5⁄6 * 1⁄6 + (5⁄6*5⁄6)^2 * 1⁄6 + ... 
       = a1(1-q^n)/(1-q) = 1⁄6 * (1-(5⁄6*5⁄6)^n)/1-(5⁄6*5⁄6)  = 6⁄11
</pre>

#### 11. Fair Coin
##### Say you flip a coin 10 times. It comes up tails 8 times and heads 2 twice. Is this a fair coin?
<pre>
Under null hypothesis that the coin is fair, probability (two-sided) of observing something more 
extreme than or equal to 8 heads and 2 tails is 
(45+10+1 (left tail) + 45+10+1 (right side))/1024 = 112/1024 > 0.05. 
So null hypothesis can’t be rejected. Hence we can’t conclude that the coin is biased.
</pre>

#### 12. Coin Flip Probability
##### Let's say you are playing a coin flip game. You start with 30. If you flip heads, you win 1, but if you get tails, you lose 1. You keep playing unitl you run out of your money (have 0) or until you win $100. What is the probability that you win $100?
<pre>
E(X) = sum(f(x)*x) = 0.5 * 1 + 0.5 * (-1) = 0.
Staring with $30, E(after one flip) = 30 + 0 = 30.
E(At the game end) is also equal to 30. In the end, we either lose and have 0 or win and have 100 Now, in the end, we will still have an expected value of $ 30 no matter what.
E(X) = 30 = p(win 100) * 100 + (1-p(win 100))* 0
=> 30 = p(win 100) * 100
=> p(win 100) = 30⁄100 = 0.3
</pre>

#### 13. Six Face Die
##### You start with a fair 6-sided die and roll it six times, recording the results of each roll. You then write these numbers on the six faces of another, unlabeled fair die. For example, if your six rolls were 3, 5, 3, 6, 1 and 2, then your second die wouldn't have a 4 on it; instead, it would have two 3s. Next, you roll this second die six times. You take those six numbers and write them on the faces of yet another fair die, and you continue this process of generating a new die from the previous one. Eventually, you'll have a die with the same number on all six faces. What is the average number of rolls it will take to reach this state?
<pre>
import matplotlib.pyplot as plt
import numpy as np
result = []
for i in range(10000):
  counter = 0
  die = [1,2,3,4,5,6]
  while len(set(die))>1:
    die = np.random.choice(die,6)
    if len(set(die))==1:
        #print(counter)
        result.append(counter+1)
        break;
    else:
        counter = counter+1
print(np.mean(result))
plt.hist(result)
plt.show()
</pre>

#### 14. Random Feed Function
##### Let's say you have a function that outputs a random integer between a minimum value, N, and maximum value, M. Now let's say we take the output from the random integer function and place it into another random function as the max value with the same min value N. 1. What would the distribution of the samples look like? 2. What would be the expected value?
<pre>
1. Uniform Distribution with [N,(N+M)/2]
2. (M + 3 * N)/4
Python Code:
N = 1
M = 10
original = np.zeros(100)
res = np.zeros(100)
for _ in range(100):
    x = np.random.randint(low = N ,high = M)
    original[_] = x
    y = np.random.randint(low = N, high = x+1)
    res[_] = y
</pre>

#### 15. Nightly Job
##### Every night between 7pm and midnight, two computing jobs from two different sources are randomly started with each one lasting an hour. Unfortunately, when the jobs simultaneously run, they cause a failure in some of the other company’s nightly jobs, resulting in downtime for the company that costs $1000. The CEO, who only has enough time today to hear no more than one word, needs a single number representing the annual (365 days) cost of this problem. Write a function to simulate this problem and output an estimated cost. How would you solve this using probability?
For the probability part, you can use geometric probability. We have a square of side length 5, x: 0 to 5 and y: 0 to 5. There are two regions of non-overlap, when y is more than 1 hour later than x and when x is more than one hour later than y. When y is more than 1 hour later than x, we have a triangle bounded by y=1+x, y=5, x=0. When x is more than 1 hour later than y, we have a triangle bounded by y=0, x=5 and y=x-1. Both of these triangles have area 0.5*4*4=8, so there is a 16⁄25 chance of non-overlap, and a 9⁄25 chance of overlap.








