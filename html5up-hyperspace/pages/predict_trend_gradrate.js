
function initGradrateGraph() {



    const data = [
        { year: 2017, score: 60 },
        { year: 2018, score: 62 },
        { year: 2019, score: 62 },
        { year: 2020, score: 62 },
        { year: 2021, score: 75 },
        { year: 2022, score: 80 }
    ];

    const margin = { top: 40, right: 20, bottom: 40, left: 40 };
    const width = 300 - margin.left - margin.right;
    const height = 200 - margin.top - margin.bottom;

    const svgGrad = d3.select('#gradrate_svg')
        .attr('width', width + margin.left + margin.right)
        .attr('height', height + margin.top + margin.bottom)
        .append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    const xScale = d3.scaleBand()
        .domain(data.map(d => d.year))
        .range([0, width])
        .padding(0.1);

    const yScale = d3.scaleLinear()
        .domain([0, d3.max(data, d => d.score)]).nice()
        .range([height, 0]);

    const xAxis = d3.axisBottom(xScale).tickFormat(d3.format('d'));
    const yAxis = d3.axisLeft(yScale);

    svgGrad.append('text')
        .attr('x', width / 2)
        .attr('y', -margin.top / 2)
        .attr('text-anchor', 'middle')
        .style('font-size', '16px')
        .text('Graduation Rate');

    svgGrad.append('g')
        .attr('transform', `translate(0, ${height})`)
        .call(xAxis);

    svgGrad.append('g')
        .call(yAxis);

    const line = d3.line()
        .x(d => xScale(d.year) + xScale.bandwidth() / 2)
        .y(d => yScale(d.score));

    const initialData = data.slice(0, 3);
    const remainingData = [{ year: 2019, score: 62 }, ...data.slice(3)];

    const initialPath = svgGrad.append('path')
        .datum(initialData)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 2);

    const remainingPath = svgGrad.append('path')
        .datum(remainingData)
        .attr('d', line)
        .attr('fill', 'none')
        .attr('stroke', 'steelblue')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .style('opacity', 0);


    const revealData = () => {
        remainingPath
            .transition()
            .duration(2000)
            .style('opacity', 1)
            .end() // Waits for the transition to complete
            .then(addDataPoints);
    };

    function addDataPoints() {
        const pointsData = [
            { year: 2019, score: 62 },
            { year: 2022, score: 85 }
        ];

        // Add circles for the data points
        // ... (the circle code remains unchanged)
        // Add circles for the data points
        svgGrad.selectAll('circle')
            .data(pointsData)
            .enter()
            .append('circle')
            .attr('cx', d => xScale(d.year) + xScale.bandwidth() / 2)
            .attr('cy', d => yScale(d.score))
            .attr('r', 4)
            .attr('fill', 'steelblue');

        // Add text labels next to the data points
        svgGrad.selectAll('.data-point-label')
            .data(pointsData)
            .enter()
            .append('text')
            .attr('class', 'data-point-label')
            .attr('x', d => xScale(d.year) + xScale.bandwidth() / 2 + 6)
            .attr('y', d => yScale(d.score) + 12) //height of label
            .text(d => d.score)
            .style('opacity', 0) //set initial opacity to 0
            .transition()
            .duration(500)
            .style('opacity', 1); //transition the opacity to 1
    }

    const revealButton = d3.select('#reveal-button-gradrate');
    revealButton.on('click', revealData);

}