import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
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


/**
 * Hello world!
 *
 */
public class Bench 
{
    public static void main( String[] args ) throws IOException
    {
        
      	try {
      		//Open rdf repository, file and connection to repository
      		Repository repo = new SailRepository(new MemoryStore());
	        File file =  new File("../rdfStore/rdfTimeStr.txt");
	        String baseURI = "http://example.org/example/local";
		repo.initialize();
		RepositoryConnection con = repo.getConnection();

		List timeVec = new LinkedList();
		int N_CYCLES = 1;
				
		con.add(file, baseURI, RDFFormat.NTRIPLES);
		List<BindingSet> resultList = null;		
		for (int i =0; i <N_CYCLES;  i ++) {
			double startTime = System.nanoTime();
			
		//	String queryString = "SELECT ?p ?w WHERE {   <http://example.org/int/0> ?w  ?p. <http://example.org/int/0> ?p  <http://example.org/int/1>} ";
		
		//	String queryString = "SELECT * WHERE {  ?s ?p  <http://example.org/int/1>.  <http://example.org/int/0> ?p  ?o} ";
		//	String queryString = "SELECT ?p WHERE {  <http://example.org/int/0> ?p  ?o} ";
		//	String queryString = "SELECT * WHERE {  ?s ?p  <http://example.org/int/1>} ";
		
			String queryString = "SELECT ?p WHERE { <http://example.org/int/666> ?p ?o.  <http://example.org/int/667> ?p  ?z} ";
			TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
		    	TupleQueryResult result = tupleQuery.evaluate();
		    	
		    	//Force evaluation or results (as suggested by sesame)

		    	resultList = QueryResults.asList(result);
		    	
		    	double time = System.nanoTime() - startTime;
		    	double duration = new Double(time / (double) 1000000);
			timeVec.add(duration);
		}
		

		
		for (BindingSet result : resultList) {
			System.out.println(result);
		}
		
		System.out.println(resultList.size());
		
		
		double mean = 0;
		double variance = 0;
		
		for (Object value : timeVec) {
			Double value2 = (Double) value;
			System.out.println(value2.doubleValue());
			mean += value2.doubleValue();
			variance += value2.doubleValue() * value2.doubleValue();
		}
		mean = mean / (double) timeVec.size();
		variance = variance / (double) timeVec.size();
		variance = variance - mean * mean;
		
		System.out.println("Variance is " + variance);
		System.out.println("Mean is " + mean);
	
	} catch (Exception e) {

		System.out.println("error :");
		System.out.println(e);
		e.printStackTrace();
	}

      
    }
}
