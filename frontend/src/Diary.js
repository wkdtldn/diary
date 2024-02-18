import React from "react";

function Diary() {
  const Save = () => {
    console.log(document.getElementById("story").value);
  };
  return (
    <div className="diaryWrapper">
      <input
        className="diaryInput"
        placeholder="Write down your own story!!"
        id="story"
      ></input>
      <button onClick={Save} className="btn-save">
        Save!
      </button>
    </div>
  );
}

export default Diary;
