This environment runs the container on AWS Batch

instance role
roles: AmazonEC2ContainerServiceforEC2Role 
trust   

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "batch.amazonaws.com",
          "ec2.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

spot fleet role (AWSServiceRoleForEC2SpotFleet)
policy:  AWSEC2SpotFleetServiceRolePolicy 

trust: 


{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "spotfleet.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

service role(AWSBatchServiceRole):
policy:  (AWSBatchServiceRole )
trust:  
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "batch.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}

