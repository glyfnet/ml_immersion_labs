import sys, random
from datetime import timedelta
from awsglue import DynamicFrame
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, TimestampType, StructType, StructField

# create the clue context
glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

def to_delta(ts, instrument):
    start = ts[0]
    mult = 10.0 if instrument == 'usdjpy' else 1000.0
    return [(x - start)*mult for x in ts]

def create_tests(ts_start, values, length, ntests):
    
    # create random set of start indexes
    start_indexes = random.sample(range(0, len(values) - length), ntests)
    
    # create new sequences for each start index of the required length
    tests = []
    for s in start_indexes:
        tests.append({
            'ts_start': ts_start + timedelta(minutes=s),
            'target': values[s: s+length]
        })   
                   
    return tests

def query_features():
    query = '''
        SELECT
            min(start) as ts_start,
            collect_list(Open) as open,
            collect_list(Close) as close,
            collect_list(High) as high,
            collect_list(Low) as low,
            collect_list(Volume) as volume
        FROM (
            SELECT 
                weekofyear(start) * 10000 + year(start) as group,
                *
            FROM (
                SELECT 
                    to_timestamp(concat(Time,"+0000"), "dd.MM.yyyy HH:mm:ss.SSSZ") as start, 
                    *    
                FROM forex 
                ORDER BY start
            )
        ) GROUP BY group
    '''
    return spark.sql(query)

def query_train_set(**kwargs):
    query='''
        SELECT 
            array({column_id},{instrument_id}) as cat,
            date_format(ts_start, 'yyyy-MM-dd HH:mm:ss') as start,
            to_delta({column}, '{instrument}') AS target
        FROM features
        WHERE ts_start < to_timestamp('{end:%Y-%m-%d}', 'yyyy-MM-dd')
        ORDER BY rand()    
    '''.format(**kwargs)
    return spark.sql(query)

def query_test_set(**kwargs):
    query='''
        SELECT 
            array({column_id},{instrument_id}) as cat,
            date_format(test.ts_start, 'yyyy-MM-dd HH:mm:ss') as start,
            to_delta(test.target, '{instrument}') AS target
        FROM (
            SELECT 
                create_tests(ts_start, {column}, {length}, {n_tests}) as tests
            FROM features 
            WHERE ts_start >= to_timestamp('{begin:%Y-%m-%d}', 'yyyy-MM-dd')  AND ts_start < to_timestamp('{end:%Y-%m-%d}', 'yyyy-MM-dd')
        )
        LATERAL VIEW explode(tests) o as test 
        ORDER BY rand()    
    '''.format(**kwargs)
    return spark.sql(query)
    
def create_all_features(instruments, columns, holdout_weeks, test_weeks, **kwargs):
    
    def write_parquet(name, df, **kwargs):
        outpath = '{s3_path}/{name}'.format(name=name, **kwargs)
        df.write.parquet(outpath, mode='append')
        print('wrote parquet file for {} to {}'.format(name, outpath))
        
    def write_json(name, df, **kwargs):
        outpath = '{s3_path}/{name}'.format(name=name, **kwargs)
        df.write.json(outpath, mode='append', timestampFormat='yyyy-mm-dd HH:mm:ss')
        print('wrote json file for {} to {}'.format(name, outpath))
        
    def union(df, new):
        return df.union(new) if df else new
    
    def shuffle(df):
        return df.orderBy(F.rand())  

    train_df = None                              
    test_df = None
    holdout_df = None 
    max_date = None
                                         
    # iterate over all instruments
    for instrument_id, instrument in enumerate(instruments):

        # create tables for each instruments csv file
        print (kwargs)
        inputpath = 's3a://{s3_bucket}/labs/deepar/data/{instrument}_1m.csv'.format(instrument=instrument, **kwargs)
        print('reading {}'.format(inputpath))
        df = spark.read.csv(inputpath, inferSchema=True, header=True).withColumnRenamed('Gmt time', 'Time')
        df.createOrReplaceTempView('forex')

        # query features into temp table   
        features = query_features()
        features.createOrReplaceTempView('features') 
        
        if max_date is None:
            max_date = features.select(F.max(features.ts_start).alias('max_date')).collect()[0].max_date
            holdout_start = max_date - timedelta(days=((holdout_weeks-1)*7))
            test_start = holdout_start - timedelta(days=test_weeks*7)
            holdout_end = max_date 

        # iterate over all columns we want to forecast
        for column_id, column in enumerate(columns):           
            # query and union new sets of data for train, test and holdout
            train_df = union(
                train_df, 
                query_train_set(
                    instrument = instrument,
                    instrument_id=instrument_id, 
                    column=column, 
                    column_id=column_id, 
                    end=test_start, 
                     **kwargs
                )
            
            )
            test_df = union(
                test_df, 
                query_test_set(
                    instrument = instrument,
                    instrument_id=instrument_id, 
                    column=column, 
                    column_id=column_id, 
                    begin=test_start, 
                    end=holdout_start, 
                    **kwargs
                )
            )
            
            holdout_df = union(
                holdout_df, 
                query_test_set(
                    instrument = instrument,
                    instrument_id=instrument_id, 
                    column=column, 
                    column_id=column_id, 
                    begin=holdout_start, 
                    end=holdout_end, 
                    **kwargs
                )
            )
                                                                                 
    # shuffle and write the test, train and holdout sets, writing them to s3
    write_json('train', shuffle(train_df), **kwargs)
    write_json('test', shuffle(test_df), **kwargs)
    write_json('holdout', shuffle(holdout_df), **kwargs)

    
# get the arguments
kwargs = getResolvedOptions(sys.argv, [
    'JOB_NAME', 
    'instruments', 'columns',
    'test_weeks','n_tests','holdout_weeks', 'length', 
    's3_bucket', 's3_path', 'user'
])

kwargs['instruments'] = kwargs['instruments'].split(',')
kwargs['columns'] = kwargs['columns'].split(',')
kwargs['test_weeks'] = int(kwargs['test_weeks'])
kwargs['n_tests'] = int(kwargs['n_tests'])
kwargs['holdout_weeks'] = int(kwargs['holdout_weeks'])
kwargs['length'] = int(kwargs['length'])
    
# register the udf
spark.udf.register("create_tests", create_tests, 
    ArrayType(
        StructType([
            StructField('ts_start', TimestampType(), True),
            StructField('target', ArrayType(DoubleType()), True)
        ])
    )
)

# register the udf
spark.udf.register("to_delta", to_delta, ArrayType(DoubleType()))

job = Job(glueContext)
job.init(kwargs['JOB_NAME'], kwargs)      
create_all_features(**kwargs)
job.commit()