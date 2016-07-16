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


/**
 * Hello world!
 *
 */
public class Test
{
    public static void main( String[] args ) throws IOException
    {
        
      	try {
//      		Open rdf repository, file and connection to repository

/*
      		Repository repo = new SailRepository(new MemoryStore());
		repo.initialize();
		RepositoryConnection con = repo.getConnection();

*/
		int FILE_LENGHT = 302924;
                File file =  new File("../rdfStore/str500D.txt");
		List<SimpleStatement> statements = new LinkedList<SimpleStatement>();
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
		}

		br.close();

		Iterable<SimpleStatement> iter = statements;
		
		List timeVec = new LinkedList();
		List timeMemVec = new LinkedList();

		int N_CYCLES = 100;
		List<BindingSet> resultList = null;
		for (int i =0; i >= 0;  i--) {
	                Repository repo = new SailRepository(new MemoryStore());
        	        repo.initialize();
	                RepositoryConnection con = repo.getConnection();

			double startMem = System.nanoTime();
			con.add(iter);

			double startTime = System.nanoTime();
			String queryString = "SELECT * WHERE {?s ?p  <http://example.org/int/100> . ?z ?s <http://example.org/int/1> . }";

			TupleQuery tupleQuery = con.prepareTupleQuery(QueryLanguage.SPARQL, queryString);
		    	TupleQueryResult result = tupleQuery.evaluate();

		    	resultList = QueryResults.asList(result);

		    	double time = System.nanoTime() - startTime;
		    	double duration = new Double(time / (double) 1000000);
			double memDuration = System.nanoTime() - startMem;
			memDuration = memDuration / (double) 1000000;
			timeVec.add(duration);
			timeMemVec.add(memDuration);
			System.out.println(resultList.size());
		}

		double mean = 0;
		double memMean = 0;
		double variance = 0;
		double memVariance = 0;

		for (Object value : timeVec) {
			Double value2 = (Double) value;
			mean += value2.doubleValue();
			variance += value2.doubleValue() * value2.doubleValue();
			System.out.println(value2);
		}

                System.out.println("Memory Time");
                for (Object value : timeMemVec) {
                        Double value2 = (Double) value;
                        memMean += value2.doubleValue();
                        memVariance += value2.doubleValue() * value2.doubleValue();
                        
               }
		
		double first = ((Double) timeVec.get(0)).doubleValue();
		mean = mean - first;
		variance = variance - first * first;
 		mean = mean / (double) (timeVec.size() - 1);
		variance = variance / (double) (timeVec.size() - 1);
		variance = variance - mean * mean;

                memMean = memMean / (double) timeVec.size();
                memVariance = memVariance / (double) timeVec.size();
                memVariance = memVariance - memMean * memMean;

		System.out.println("Activation time is :" +  first);

		System.out.println("Execution Variance is " + variance);
		System.out.println("Execution Mean is " + mean);

                System.out.println("Memory Variance is " + memVariance);
                System.out.println("memory Mean is " + memMean);



	} catch (Exception e) {

		System.out.println("error :");
		System.out.println(e);
		e.printStackTrace();
	}


    }
}
