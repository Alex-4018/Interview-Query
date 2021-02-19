### 1. How would you determine how to rank posts in the newsfeed?

C: Okay, Are we discussing facebook newsfeed here or some other SN?
C: By Posts, i am inferring anything that appears on the newsfeed like photos, videos, memes, links, Ads etc.
C: By ranking posts in the news feed, i am assuming that we focus on the order of posts in the news feed,ie, what appears first and what later. Am i correct?
C: Okay.Are we focussed on the newsfeed of a stable, old user or a new user cause the news feed may look different in the two cases?
Now the User group is a stable, old User of Facebook.
Although there can be different personas of the users but we might want to make an algorithm that fits well with most of the users.
We may change the algorithm a little bit depending upon the User Groups and geography/demography and this can be a part of our experiments.

Now, lets first think what are the factors that determine the importance of the post to me as a user.

1. How close/good friend am i to the person whose post we have to rank?
I will want to see the post of a person who is a good friend in real life first in my news feed. The closest proxy to real life friendship is facebook engagement,ie if i engage with someone's posts regularly i am a close friend of that person and should be ranked higher in my news feed.

2. How important is that post?
Posts which are of an important life event like Convocation, Marriage etc. are very important posts and they should be propelled to the top of news feed as these signify an important life event.

3. What is the post type? Video/ link/ text/photo etc
Now, facebook has the data that as user, I like to engage more with a particular type of post, which maybe  videos or memes or links. Showing them first will capture more attention of the user and facebook would want to show them  first.

4. Is the post Recent/old?
Recent posts are more likely to be relevant as on today. Hence, it makes sense to show recent posts first. However, this should be seen in conjunction with other variables like Posts importance.

5. How much other people have liked/reacted to the post?
More engagement per post means more is the quality of that post. Its like a real time crowdsourced way to tell about a post's quality. Hence, the post that gets more likes gets a better Post Rank

6. For Ads, how relevant is the Ad to me as a User?
Now, showing Ads between posts is as much an Art as it is a science.
Lot of Ads will drive away the user. Very few Ads will mean less Revenue.
The important thing here is Relevance of the Ad to the User and his past Ad interactions.
Ads are less suitable for top of the news feed. They should be placed in between organic posts. But it is important to place Relevant Ads( coming from User's browsing behavior) shown first then other ads.

 
Now, the actual relative importance of the above logics can be part of an A/B/N experiment that aims to maximise user engagement and facebook revenue.
The relative weights may also depend on the demographic and country of the user. Eg. Certain country users enagage with Ads or Videos more than other countries etc.
This will an algorithm that will be always a work in progress and evolve itself to suit the goals of engagement and revenue.

### 2. You are the PM for a streaming video service. You come into the office and see that one key metric has dropped by 80%. 
80% is a huge drop in any key metric.
I would try to first narrow down exactly what metric this is so I would ask the interviewer if they could tell me if the metric is new user retention, churn, monetization etc.
Second, I would try to understand if this was a sudden or gradual drop. for 80% definitely sounds like a sudden drop or else somebody would’ve said something already.
If it’s a sudden drop, I would try to pinpoint around what time this drop occured and figure out if there were any internal/external factors that could have caused it.

internal factors include: new feature was released, server went down, a new bug became prevalent. For the last two, you can segment it by region, browser/device type, and OS type. The issue could also be that the metrics we are grabbing is incorrect.

external factors include: a new competitor has joined into the market, bad PR, maybe a firmware was pushed outside of your control. It could also be due to seasonality or a major temporary event. If it’s a major temporary event, you should see KPIs begin to return to their normal state shortly.

Third, I would try to see if any other relational KPI drops. It’s easier to know what KPI it is before, but we can go along the user journey and see if any KPI before it dropped.

IE: A user signs up for the service -> enters in a credit card for payment (optional) -> clicks on a video to watch -> Watches the video -> chooses another video to watch
This is important in narrowing down exactly when the problem first starts. For example, if a key KPI is number of videos watched, perhaps the sign in is where most people are failing.

If the issue is a feature, I would try to clarify what the goal of the feature is. It could be possible that we started doing targeted ads and conversion dropped but the first time purchase after clickthrough increased. It would be important to understand if the goal of the feature change was met even with this big of a KPI drop.

If I can ascertain the exact issue, I would work with Sys Ops, Engineering, and other people on my team to try to address it. If the issue is a bug, we would have to issue a hotfix. If the issue is a server, than sys ops can look into it. If the issue was due to a feature release, we should probably look into either fixing it or reversing it quickly.

If the issue is external, this would be harder to solve immediately and would often require going through the normal cycle of product development to address them.

So in summary, I would first make sure we can ascertain if the drop was temporary or permanent, gradual or sudden and if the KPI drop may have occured elsewhere int he user funnel. I would look at internal and external factors to try to see if I can pinpoint the issue. Third, if the issue can be fixed immediately, I would contact my team to put out a hotfix or roll back a change that we may have made. If not, we should understand the issue thoroughly before acting and let people in the company know of our findings.

### 3. How you would evaluate the effect on engagement of teenage users when their parents join Facebook?
Since you cannot run a randomized test (unless you figure out a way to make parents of teens to join by force), this will be need to be an observational study, with a quasi experiment design to answer the question - ‘How do parents cause teens to behave in different ways’.

Look at 2 groups of teen users at two time periods. Time period t0, parents of teens in group 1 (test) join facebook while parents in group2 (control) do not. At time period t2, compare the pre to post change in user behavior of users in test to that of control.

Since random assignment is not possible, you’ll need to control for selection bias through matching or regression. Variables to match on would depend on the outcome measure of interest (time spent, engagement on tagged posts, sharing, posting). Few selection controls could include age, affluence, education level of teens and parents, ethnic/cultural background, size and density of connections etc.

Ultimately, compare the pre to post change in metrics for the 2 groups at time t1 (relative to time t0) and see if the differences are significant.


### 4. How would you test whether having more friends now increases the probability that a Facebook member is still an active user after 6 months?
We can build a supervised machine learning model such as a binary classifier to model if a facebook member is still active after 6 months.

Target: * pos class - member is active after 6 months * neg class - member is not active after 6 months

We need to pick a model that would output probability of the pos and neg class. Logistics Regression will output the log-odds coefficients.

The interpretation is that if the log-odd for the number of friends is greater than 0, we can say having more friends will increase the probability of the member being active after 6 months.

### 5. Build a restaurant product for Facebook 
Seeking Clarification:

1, Restaurant Product as in product that allows Restaurants to connect with their customers?

2, Is the Online Ordering platform?

3, Is this a reservation app, where customers can make an appointment at the restaurant?

4, Is this a product/app where users can provide their feedback and rate restaurants?

5, What are some of the objectives for this Product?

6. Is this product vision to leverage the existing Facebook ecosystem for the Product/bundled with FB or are we proposing a standalone product?

Interviewer: The vision is to bundle the product with Facebook and the main objective is to connect Restaurants with their customers

That sounds great, since Facebook's mission is to connect people and bring the world closer, the objective of the restaurant app aligns with the broader mission of the company.

I will structure my answer using a simple framework:

I will first brainstorm objectives and goals, then talk about the user segments, discuss pain points and problems we should be solving for these users, if we come up with too many pain points and problems I would like to prioritize them and focus on the top two or three, I will then like to propose some solution, dram a simple wireframe and finally suggest some metrics that will allow us to measure if the Product is achieving the goals that we have established

Objectives & Goals

Most businesses operate with three main goals - User acquisition, User engagement and Monetization

Since this is a brand-new product (app) I would like to focus on the first two objectives that is to acquire user base both for the new restaurant product as well as FB. Also, since this app/product will act as an extension of Facebook it will also improve user engagement as FB users (both individuals and restaurants) who currently use some other restaurant products like Yelp or OpenTable can perform similar activities on the FB platform and expand their social presence.

User Segments

There are two main Users Personas that the product will cater to:

Restaurants & Customers

Now within the Restaurant segment you could have high end/Michelin rated/formal restaurants that are less frequented by the general population, then you have your mid-market and casual restaurants that are most commonly used by people and finally there are small/ mom & pop/family owned restaurants that specialize in certain cuisines and fall in the mid to low price range.

In the customer segment too, you could have fancy diners, who are exploring gourmet cuisines or celebrating special occasions and then you have the casual diners who are looking to enjoy a meal in a very casual setup and given the COVID situation the third type of customers that prefer to take out or have food delivered to their doorstep.

From the customer segments I will focus on the mid-range and local/ family owned restaurants since they both have similar set of requirement, and on the customer side l I will focus on the casual diners for our first version of the product/app.

Problems/Pain Points 

Problem/Pain Points for Restaurants:

Outreach, spreading awareness, gaining market share, building relationships to promote repeat customers, targeted marketing

Specially in the COVID situation convey their preparedness to the customers by ensuring they are following all protocols, providing for contactless carryout service wherever possible

With food pickup & delivery being rampant in the COVID situation, be able to enable online ordering and payments, provide faster delivery, not mixing up orders and basically streamline the entire delivery and pickup process

Problem/Pain Points for Users:

Locating restaurants, getting recommendations on food, look for deals

If dining-in then make reservations

In current environment be able to browse menu online, order online and schedule either delivery or pickup at a desired time.

At a broader level those are the pain points I would like to focus on, would you like me to consider any other specific problems?


Solution

In terms of solution, I would like to suggest a new Market Place for Restaurants. 

Restaurants will be able to create their own workspaces or FB pages. The Market place landing page will consider user's location to default the most popular, most reviewed restaurants in his or her neighborhood. The page will also have a search options where users can search based on one or more of the following attributes:
By Cuisine
By Price range
By what is currently open
By which the one's frequented by my Friends (since this built on top of the FB ecosystems, we should be locate the frequented or visited restaurants when user's friends check- in)'
By friend's recommendation
By ratings
By the one's currently running promotions
By Name
By Zip Code/City/Location
Users will be able to click on their preferred restaurant and navigate the Restaurant’s FB page where they will have the ability to reserve a table
For restaurants with friends check-ins or reviews the system will prioritize these and display at the top of the restauran page
Access Menu, order online and checkout, during checkout a prompt will allow user to publish a checking to his or her FB page to enable promotion. Restaurants can incentivize their users for publishing their check-ins
Users can also schedule their pickup or delivery for a specific time
Restaurants can use the standard FB pages to build marketing content, deals, pictures and given the current situation also create videos with a live tour of their kitchen and masked/gloved employees
Wireframe

Prioritization – I will prioritize the following for the first Beta Testing/prototype
Market place for Restaurants
Discovery for Restaurants based on location, and all other attributes mentioned above especially connection activity. Connection activity would be top on my list cause that will allow users to receive tailored recommendations/feedback

Key Metrics:
DAU for Marketplace
Number of Restaurants joining Market place week over week
Number of Check-ins by Users

### 6. The online price is dependent on the availability of the product, the demand, and the logistics cost of providing it to the end consumer. You discover our algorithm is vastly under-pricing a certain consumer product. What are the steps you take in diagnosing the problem?

Traditional price optimization requires knowing or estimating the dependency between the price and demand. This basic model can be further extended to incorporate item costs, cross-item demand cannibalization, competitor prices, promotions, inventory constraints and many other factors. The traditional price management process assumes that the demand function is estimated from the historical sales data, that is, by doing some sort of regression analysis for observed pairs of prices and corresponding demands. Since the price-demand relationship changes over time, the traditional process typically re-estimates the demand function on a regular basis.
- If the product life cycle is relatively long and the demand function changes relatively slowly, the passive learning approach combined with organic price changes can be efficient, as the price it sets will be close to the true optimal price most of the time.
- If the product life cycle is relatively short or the demand function changes rapidly, the difference between the price produced by the algorithm and the true optimal price can become significant, and so will the lost revenue. In practice, this difference is substantial for many online retailers, and critical for retailers and sellers that extensively rely on short-time offers or flash sales (Groupon, Rue La La, etc.).

### 7. Let's say you're tasked with building a classification model to determine whether a customer will buy on an e-commerce platform after making a search on the homepage. You find that your model is suffering from low precision. How would you improve it?

Majority of the customers coming on e-commerce site do not commit on buying the same instant and are window shopping, this gives us a hint that this will be an imbalanced class. Some of the features, we can use would be the month in which sales are made, as customers tend to convert and buy in some months than other. We have to data exploration to see this Also, we could see, what channels a customer is coming from, customer coming from one channel might have higher probability of converting than another. We could also see, if its a returning customer or a new customer. What’s the log in style of the customer: logged in as user or guest Whats the average yearly, monthly purchase value for thia customer

### 8. Suppose we are training a binary classification algorithm. 99.8% of individuals in our sample have a value of 0 and 0.2% have a value of 1. We downsample and we randomly sample say 1% of the individuals which have a value of 0, but keep all other individuals with a value of 1. Now we re-train the model on the smaller sample and build a binary classifier that predicts the probability of an individual with a value of 1. How would we then adjust our output probabilities to use this model on the total population?
I think this problem should be medium max, not hard. Binary classifier gives a float output between 0 and 1 and you have to assign True or False based on a threshold, usually 0.5. Just tweak the threshold so that the test set T/F ratio is similar to that of the original train set. In this case the threshold would probably be around 0.99 or higher, but you could test each threshold in a loop.

### 9. Let's say you're given 90 days of ride data.How would you use it to project the lifetime of a new driver on the system? What about the lifetime value from the driver?
Lifetime 1. Survival analysis. At any point in time t, we would calculate the probability of the driver dropping off. 2. Find out what makes drivers different from those who stay to those who leave by week X. For example, did they ride a lot or not so often, did they go for the new driver promotions. We could also break it down by product: Line, Lyft, and Black/SUV. Each of them will behave differently and they also make different amounts of money. Line rides are cheaper but you also make more of them at a time. 3. We should also break it down by city. 4. Breaking down helps and we can later do some averaging.
Lifetime value 1. Is 90 days enough to project lifetime value? I am not so sure about that… Ask if we can get data for a longer period of time depending on how likely a driver is to stay after 90 days. e.g. the %drivers that still drive at 90 days out of those that started in the same cohort. 2. Breakdown by type of service, cities, etc.

### 10. Let's say we want to build a model to predict the time spent for a restaurant to prepare food from the moment an order comes in until the order is ready. What kind of model would we build and what features would we use?

This is a regression problem. It’s probably not linear as it depends on many factors. There are some clear and intuitive predictors which are the number of staff in the kitchen, number of clients in the restaurant or number of orders waiting. You need some way of characterising the efficiency of the staff in the kitchen. You can do that by measuring the ammount of dishes served per minute for each of the staffs in the kitchen and use that as a predictor for each one. Same for the clients. A client that just entered is very likely to order a dish in a few minutes while a client already eating and that has been in the restaurant for a long time won’t have any effect on the kitchen. You could characterise that by having how much time each of the clients has been in the restaurant. Another thing to consider is that there could be 0 to N clients in the restaurante and 0 to N workers in the kitchen so missing predictors will have to be filled with a logical value e.g. we have space for N clients in our restaurant. We will have N predictors for each of the N clients. What if we have only half of these N clients?. What will we do with the other half of the predictors? One more predictor to take into account is the type of dishes that are going to be pending to be cooked. The waiting time is not going to be the same if we have 5 mains waiting to be cooked compared to 5 cold appetisers

### 11. Let's say we have 1 million app rider journey trips in the city of Seattle. We want to build a model to predict ETA after a rider makes a ride request. How would we know if we have enough data to create an accurate enough model?
Question is definitely unanswerable without knowing the business metrics. What is “accurate enough” for the business? I would definitely mention train/test/validate datasets (maybe 70%/2.0%/10%), but I think there’s a way to guess whether a model will actually work before building it.

Could try to fit distributions to the ETAs as functions of individual variables (holding others constant) and seeing what families you get. If distributions are normal-ish (i.e. from distributions that quickly converge), you’re likely safe. If distributions look more like power laws, you may need more data.

We need to ask how much error in ETA is acceptable. If I am not provided with error that company uses I would use rmse. We need to do cross validation on our data. If the rmse is not deviating a lot across folds that means the model has generalised well and we have enough data.


Say we set aside 30% for test and train with 70% , i.e 700 K data points and we observe a reasonable metric (say regression R2 is over 80%) on train and test data, then I think thats a reasonable premise to say that we have enough data?

On the other hand if the metric is poor say 60% R2 score and the learning-rate ( performance vs training set size) does not show any signs of saturating, that may be a reason to seek more training data. We may have to reduce dimensionality as well before experimenting other stuff.

### 13. Let's say you have to build scrabble for Spanish users.Assuming that you don't know any Spanish, how would you approach assigning each letter a point value?
The bag of letter in scrabble might have more pieces of frequently used letter and less pieces of rarely used letter. We can just assign rarely values large points and frequent values low points. May be we can get probability/frequency of each letter then we can invert it and may be round it to an integer.
Other approach is to watch many scrabble games and see which letter is used most of the times by users and make a distribution based on usage. Now you again have usage frequency of letters , frequently used words will have higher number and rarely used words will have lower. So just invert the usage frequencies of letters and round to nearest integer.

Let’s assume we have access to:
A large Spanish corpus
A large English corpus
Scrabble letter scores for English

For each corpus, filter out the following:
One-letter words
Proper nouns (anything with a capital letter that doesn’t follow a period; alternatively, anything not in a dictionary, although we may not have access to a dictionary)
Contractions
After that, look at the frequency of letters. See how they compare with letter scores for English, then follow roughly the same pattern for Spanish.

### 14. You're tasked with building a model to figure out the most optimal way to send ten emails copies that the content team generated to increase conversions to a list of subscribers.

Summary steps:

Understand what each email that is being sent talks about, understand the content of it. If the email looks long, try to describe the email in 4-5 words.

Assuming that the list of subscribers to target to is already given, group the similar kind of subscribers into one - an unsupervised clustering algorithm should work to group similar subscribers into one.

Assuming, that we have the user activity of the list of subscribers from which we can get their interest, find the common interest words within each cohort created from above and match it with the email content using cosine similarity between the two - I.e interests of a cohort and the content of the email. The cohort with the highest cosine similarity to the email should be sent that particular email.
