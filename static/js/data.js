
        const dataIndexes = [];

        document.querySelectorAll('.controlButton').forEach(button => {
            button.addEventListener('click', () => {
                const dataIndex = button.getAttribute('data-index');
                dataIndexes.push(dataIndex);

                // Send the dataIndexes array to the server
                fetch('/store_data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ dataIndexes: dataIndexes })
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Success:', data);
                })
                .catch((error) => {
                    console.error('Error:', error);
                });
            });
        });
