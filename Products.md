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
