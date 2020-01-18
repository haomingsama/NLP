-- Given a member_id of a member, find out the top 10 posts which are: 
-- 1. The most voted among people he followed, and
-- 2. Never voted on by the member himself, and
-- 3. Having at least 1 vote from anybody within the past 24 hours, and 
-- 4. Not created by the member himself, and
-- 5. Created within recent 3 days

-- Assume member id  =1
-- Database name : Interview

select post_id, count(member_id) as hot from interview.vote
where
	post_id  in
	(select post_id from interview.post
		where creator_id in
			(select  followed_id from interview.following
				where interview.following.follower_id = 1))    -- Task 1
	
	and post_id not in
		(select post_id from interview.post
		where creator_id =1)                                              -- Task 2
	
    and post_id in (select post_id from interview.post
		where to_days(now()) - to_days(created_at) <=3)   -- Task5
	
    and voted_at > DATE_SUB(NOW(),INTERVAL 24 HOUR)  -- Task 3
	
	and member_id !=1  -- Task5
group by post_id
order by hot desc                                                               -- Ouput top 10 posts
limit 10
