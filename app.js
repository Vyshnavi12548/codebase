// Inside your fetchData function in the React App.js file
const fetchData = async () => {
    setIsLoading(true);
    setCompanyData(null); 
    
    try {
        const response = await fetch(`http://127.0.0.1:5000/api/score?company=${companyName}`);
        if (!response.ok) {
            throw new Error('Company not found or API error.');
        }
        const data = await response.json();
        setCompanyData(data);
        setIsLoading(false);
    } catch (error) {
        console.error("Failed to fetch data:", error);
        setIsLoading(false);
    }
};