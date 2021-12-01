import json
import boto3
from datetime import datetime

sage_client = boto3.client("sagemaker")

def lambda_handler(event, context):
    
    current_time = datetime.now().strftime('%Y-%m-%d--%I-%M-%p')
    
    # TODO implement
    pipeline_response = sage_client.start_pipeline_execution(
        PipelineName='RetrievalPipeline',
        PipelineExecutionDisplayName=f'Lambda-Execution-{current_time}',
        PipelineParameters=[
            {
                'Name': 'ProcessingInstanceType',
                'Value': 'ml.m5.xlarge'
            },
            {
                'Name': 'TrainingInstanceType',
                'Value': 'ml.m5.2xlarge'
            }, 
            ],
        PipelineExecutionDescription='Pipeline-Executed-From-Lambda',
    )
    
    return {
        'statusCode': 200,
        'body': json.dumps({
                'pipeline_response': pipeline_response['ResponseMetadata']['HTTPStatusCode'],
                'pipeline_arn': pipeline_response['PipelineExecutionArn'],
                'pipeline_execution_date': pipeline_response['ResponseMetadata']['HTTPHeaders']['date']
            })
    }
