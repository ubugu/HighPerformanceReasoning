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
public class App 
{
    public static void main( String[] args ) throws IOException
    {
        Repository repo = new SailRepository(new MemoryStore());
        File file =  new File("../rdfStore/rdf2.txt");
        String baseURI = "http://example.org/example/local";
        //copy();
        try {
		repo.initialize();
		RepositoryConnection con = repo.getConnection();

		List timeVec = new LinkedList();
		int N_CYCLES = 50;
		

		try {
			con.add(file, baseURI, RDFFormat.NTRIPLES);
			
			

		        long startTime = System.nanoTime();
			String queryString = "SELECT ?p ?y WHERE { <http://example.org/int/0> ?p ?y . <http://example.org/int/0> ?p ?y } ";
			TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
		    	List<BindingSet> resultList;
		    	TupleQueryResult result = tupleQuery.evaluate();
		    	resultList = QueryResults.asList(result);
		    	long endTime = System.nanoTime();
			Long duration = new Long((endTime - startTime) / 100000);
//		    	System.out.println(i);
			System.out.println(duration.longValue() );
			timeVec.add(duration);
			
			
			
			long mean = 0;
			long variance = 0;
			for (Object value : timeVec) {
				
				Long value2 = (Long) value;
				System.out.println(value2.longValue());
				mean += value2.longValue();
				variance += value2.longValue() * value2.longValue();
			}
			mean = mean / timeVec.size();
			variance = variance / timeVec.size();
			variance = variance - mean * mean;
			System.out.println("Variance is " + variance);
			System.out.println("Mean is " + mean);

	   	}   catch (Exception e) {

	   		System.out.println("inner error");
	   		System.out.println(e);
	      	}
		

	} catch (Exception e) {

		System.out.println("outer error");
		e.printStackTrace();
	}

        System.out.println("Funza");
    }
}
