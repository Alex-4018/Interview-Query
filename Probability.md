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

