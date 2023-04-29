import React from 'react';
import Answers from './Answers';

function Results({ data }) {
  let questions = [];

  const cleanData = () => {
    questions = data.split(/\d(?=\.)|\n|Q/).filter(Boolean);
    // let regex = /^(\d+\.\s|A:\s)/;
    // let questions = data.split(regex).filter(Boolean);
  };

  cleanData();
  return (
    <>
      <h1 className="questions_title">Practice Questions</h1>
      <hr />
      {questions.map((x, indx) =>
        x[0] === '.' || x[0] === ':' ? (
          <h4 className="questions_question" key={indx}>
            {x.slice(1)}
          </h4>
        ) : (
          <Answers key={indx} answers={x} />
        )
      )}
    </>
  );
}

export default Results;
