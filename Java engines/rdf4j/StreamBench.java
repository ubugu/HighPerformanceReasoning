import org.eclipse.rdf4j.model.*;
import java.nio.channels.*;
import org.eclipse.rdf4j.model.impl.*;
import java.io.*;
import java.net.URL;
import java.util.*;
import java.lang.Object;
import org.eclipse.rdf4j.model.Value;
import org.eclipse.rdf4j.model.ValueFactory;
import org.eclipse.rdf4j.query.BindingSet;
import org.eclipse.rdf4j.query.QueryLanguage;
import org.eclipse.rdf4j.query.QueryResults;
import org.eclipse.rdf4j.query.TupleQuery;
import org.eclipse.rdf4j.query.TupleQueryResult;
import org.eclipse.rdf4j.repository.Repository;
import org.eclipse.rdf4j.repository.RepositoryConnection;
import org.eclipse.rdf4j.repository.sail.SailRepository;
import org.eclipse.rdf4j.rio.RDFFormat;
import org.eclipse.rdf4j.sail.memory.MemoryStore;
import mydomain.CircularBuffer;	
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.nio.file.Files;


/**
 * Hello world!
 *
 */
 

 
public class StreamBench
{





    public static void main( String[] args ) throws IOException
    {

      	try {
      		if (args.length < 3) {
      			System.out.println("ERROR INCORRECT INPUT VALUES; THE VALUES ARE: QUERY, WINDOW, STEP");
      			return; 
      		}
      		
                File file =  new File(args[3]);
		List<SimpleStatement> statements = new ArrayList<SimpleStatement>();
		List<Long> timestamps = new ArrayList<Long>();
		
		//Reading input from file to main memory
		FileInputStream fis = new FileInputStream(file);
		BufferedReader br = new BufferedReader(new InputStreamReader(fis));
		ValueFactory factory = SimpleValueFactory.getInstance();
		String line = null;
		while ((line = br.readLine()) != null) {
			String[] parts = line.split(" ");
 			parts[0] = parts[0].substring(1, parts[0].length() - 1);
                        parts[1] = parts[1].substring(1, parts[1].length() - 1);
                        parts[2] = parts[2].substring(1, parts[2].length() - 1);

                        IRI sub = factory.createIRI(parts[0]);
                        IRI pre = factory.createIRI(parts[1]);
			IRI obj = factory.createIRI(parts[2]);
			SimpleStatement statement = (SimpleStatement) factory.createStatement(sub, pre, obj);
			statements.add(statement);
			
			if (parts.length == 5) {                                                                                                                                                                                                    
				timestamps.add(Long.parseLong(parts[4].substring(0, parts[4].length() - 3)));

				
			}			
		}
		br.close();
	
		List timeVec = new LinkedList();
		List queryVec = new LinkedList();

		int N_CYCLES = 2;
		List<BindingSet> resultList = null;
		int resultLen = 0;
		for (int i =0; i < N_CYCLES;  i++) {

			LinkedList<SimpleStatement> currentStm = new LinkedList<SimpleStatement>();
			
			int bufferSize = 400000;		
			CircularBuffer<Long> timestampBuffer = new CircularBuffer<Long>(bufferSize, new Long[bufferSize]);
			Long  currentTimestamp = timestamps.get(0) - 1;
	
			int windowTime = Integer.parseInt(args[1]);
			int stepTime = Integer.parseInt(args[2]);
			
			
			for (int k = 0; k < statements.size(); k++)
			{
				timestampBuffer.buffer[k % bufferSize] = timestamps.get(k);
				timestampBuffer.end = k % timestampBuffer.size;
								
				/*if ( timestampBuffer.buffer[k % bufferSize] > currentTimestamp +  windowTime ) {
		
					while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
						timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
						currentStm.removeFirst();
					}
					
					Repository repo = new SailRepository(new MemoryStore());
					repo.initialize();
					RepositoryConnection con = repo.getConnection();	
					con.add(currentStm);
					String queryString = args[0];
					TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
				    	TupleQueryResult result = tupleQuery.evaluate();
				    	resultList = QueryResults.asList(result);
										
					resultLen += resultList.size();                      
					
					currentTimestamp += stepTime;
					
					for (int zw = 0; zw < resultList.size(); zw++) {				
						System.out.println(resultList.get(zw));
					}

					
					System.out.println("FOUND " + resultList.size());

				}*/
				currentStm.add(statements.get(k));
				
			}
			
			
			if ( currentStm.size() != 0) {
				
				while(timestampBuffer.buffer[timestampBuffer.begin] <= currentTimestamp) {
					timestampBuffer.begin = (timestampBuffer.begin + 1) % timestampBuffer.size;
					currentStm.removeFirst();
				}
				Repository repo = new SailRepository(new MemoryStore());
				repo.initialize();
				RepositoryConnection con = repo.getConnection();	
				con.add(currentStm);
				String queryString = args[0];
				
				double queryStart = System.nanoTime();
				TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
			    	TupleQueryResult result = tupleQuery.evaluate();
			    	resultList = QueryResults.asList(result);
				double innerEnd = System.nanoTime();
				double queryDuration = new Double ((innerEnd - queryStart) / (double) 1000000);
				queryVec.add(queryDuration);
				
				
				resultLen += resultList.size();
				currentStm.clear();
				
				currentTimestamp += stepTime;
				
				System.out.println("FOUND " + resultList.size());
			}


		}
		

		queryVec.remove(0);
		
		Double mean = new Double(getMean(queryVec));
		
		for (int i = 0; i < queryVec.size(); i++) {
			System.out.println(queryVec.get(i));
		}
		
		
                System.out.println("total res are " + resultLen );
                
                
		Files.write(Paths.get("./result.txt"),  (mean.toString() + " \n").getBytes(), StandardOpenOption.APPEND);
	} catch (Exception e) {

		System.out.println("error :");
		System.out.println(e);
		e.printStackTrace();
	}
	

    }
    
    
    	public static double getMean(List elements) {
		double mean = 0;
                for (Object value : elements) {
                        Double value2 = (Double) value;
                        mean += value2.doubleValue();
                }
		mean = mean / (double) elements.size();
		return mean;
	}
	
	public static double getVariance(List elements) {
		double mean = 0;
		double variance = 0;
                for (Object value : elements) {
                        Double value2 = (Double) value;
                        mean += value2.doubleValue();
                        variance += value2.doubleValue() * value2.doubleValue();
                }
                
                mean = mean / (double) elements.size();
                variance = variance / (double) (elements.size());
                variance = variance - mean * mean;

		return variance;
	}
}
